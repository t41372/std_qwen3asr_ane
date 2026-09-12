// Bounded read-only IOReport probe. Private API declarations follow macmon's
// src_lib/sources.rs; see research/nonroot-power.md for provenance and caveats.
// Build: clang -O2 -framework Foundation -lIOReport experiments/nonroot_power_probe.m -o artifacts/power-probe/nonroot_power_probe
#import <Foundation/Foundation.h>
#include <time.h>
#include <unistd.h>

extern CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
extern CFTypeRef IOReportCreateSubscription(void *, CFMutableDictionaryRef,
                                           CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(CFTypeRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);

static double monotonicSeconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec / 1e9;
}

static NSString *label(CFStringRef value) {
    return value ? [(NSString *)value stringByTrimmingCharactersInSet:
                    [NSCharacterSet whitespaceAndNewlineCharacterSet]] : @"";
}

static void emit(NSDictionary *record) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        int count = argc > 1 ? atoi(argv[1]) : 5;
        if (count < 1 || count > 120) return 2;
        CFDictionaryRef all = IOReportCopyAllChannels(0, 0);
        if (!all) { fprintf(stderr, "IOReportCopyAllChannels failed\n"); return 1; }
        NSMutableDictionary *selected = [(NSDictionary *)all mutableCopy];
        NSMutableArray *channels = [NSMutableArray array];
        for (NSDictionary *item in selected[@"IOReportChannels"]) {
            CFDictionaryRef channel = (CFDictionaryRef)item;
            NSString *name = label(IOReportChannelGetChannelName(channel));
            if ([label(IOReportChannelGetGroup(channel)) isEqual:@"Energy Model"] &&
                ([name hasSuffix:@"CPU Energy"] || [name isEqual:@"GPU Energy"] ||
                 [name hasPrefix:@"ANE"] || [name hasPrefix:@"DRAM"])) {
                [channels addObject:item];
            }
        }
        selected[@"IOReportChannels"] = channels;
        CFMutableDictionaryRef subscribedChannels = NULL;
        CFTypeRef subscription = IOReportCreateSubscription(NULL, (CFMutableDictionaryRef)selected,
                                                           &subscribedChannels, 0, NULL);
        if (!subscription) { fprintf(stderr, "IOReport subscription failed\n"); return 1; }
        emit(@{@"event": @"subscription", @"uid": @(getuid()), @"channel_count": @(channels.count)});
        CFMutableDictionaryRef sampleChannels = subscribedChannels ?: (CFMutableDictionaryRef)selected;
        CFDictionaryRef previous = IOReportCreateSamples(subscription, sampleChannels, NULL);
        double previousTime = monotonicSeconds();
        if (!previous) return 1;
        for (int index = 0; index < count; index++) {
            sleep(1);
            CFDictionaryRef next = IOReportCreateSamples(subscription, sampleChannels, NULL);
            double nextTime = monotonicSeconds();
            if (!next) return 1;
            CFDictionaryRef delta = IOReportCreateSamplesDelta(previous, next, NULL);
            if (!delta) return 1;
            double duration = nextTime - previousTime;
            NSMutableArray *readings = [NSMutableArray array];
            for (NSDictionary *item in ((NSDictionary *)delta)[@"IOReportChannels"]) {
                CFDictionaryRef channel = (CFDictionaryRef)item;
                NSString *unit = label(IOReportChannelGetUnitLabel(channel));
                int64_t raw = IOReportSimpleGetIntegerValue(channel, 0);
                double scale = [unit isEqual:@"mJ"] ? 1e-3 : [unit isEqual:@"uJ"] ? 1e-6 :
                               [unit isEqual:@"nJ"] ? 1e-9 : 0;
                NSMutableDictionary *reading = [@{@"channel": label(IOReportChannelGetChannelName(channel)),
                    @"unit": unit, @"raw_delta": @(raw)} mutableCopy];
                if (scale && raw >= 0) {
                    reading[@"joules"] = @(raw * scale);
                    reading[@"watts"] = @(raw * scale / duration);
                }
                [readings addObject:reading];
                [reading release];
            }
            emit(@{@"sample": @(index), @"duration_s": @(duration),
                   @"start_monotonic_s": @(previousTime), @"end_monotonic_s": @(nextTime),
                   @"channels": readings});
            CFRelease(delta);
            CFRelease(previous);
            previous = next;
            previousTime = nextTime;
        }
        CFRelease(previous);
        CFRelease(subscription);
        if (subscribedChannels) CFRelease(subscribedChannels);
        [selected release];
        CFRelease(all);
    }
    return 0;
}
