#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#import "bindings.h"

int main(int argc, char * argv[]) {
    run_app();
}
