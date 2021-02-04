use ocean::run;

#[no_mangle]
pub extern "C" fn run_app() {
    crate::run();
}
