fn main() {
    println!("cargo:rerun-if-changed=vendor/shim.cpp");
    println!("cargo:rerun-if-changed=vendor/pocketfft_hdronly.h");
    if std::env::var("CARGO_FEATURE_POCKETFFT").is_err() {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("vendor/shim.cpp")
        .include("vendor")
        // Upstream defaults the plan cache to 0, which would re-plan every frame.
        .define("POCKETFFT_CACHE_SIZE", "16")
        // pocketfft's own threading would nest inside rayon's; keep it single.
        .define("POCKETFFT_NO_MULTITHREADING", None)
        .opt_level(3)
        // Do NOT add -ffp-contract=off here. The compiler fusing multiply-adds into
        // FMAs is load-bearing: PyTorch's own pocketfft build contracts, and turning it
        // off moves the worst difference from exactly zero to 4e-16 (f64) and 2e-07
        // (f32). SLEEF's sources ask for -ffp-contract=off, which is what makes this
        // worth writing down — that instruction applies to SLEEF, not here, and this
        // crate's SLEEF port sidesteps it entirely by using explicit `mul_add`.
        .warnings(false)
        .compile("rustft_pocketfft");
}
