pub trait DiagnosticCheck {
    fn name(&self) -> &'static str;
    fn run(&self) -> bool;
}

pub fn run_all(checks: &[Box<dyn DiagnosticCheck>]) -> bool {
    let results: Vec<(&'static str, bool)> = checks
        .iter()
        .map(|check| {
            let pass = check.run();
            println!();
            (check.name(), pass)
        })
        .collect();

    println!("\n================= SUMMARY =================");
    for (name, pass) in &results {
        println!("{:<32} {}", name, if *pass { "PASS" } else { "FAIL" });
    }
    results.iter().all(|(_, pass)| *pass)
}
