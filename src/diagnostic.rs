pub trait DiagnosticCheck {
    fn name(&self) -> &'static str;
    fn run(&self) -> bool;

    fn log(&self, msg: &str) {
        println!("  {msg}");
    }
}

pub struct DiagnosticSuite {
    checks: Vec<Box<dyn DiagnosticCheck>>,
}

impl DiagnosticSuite {
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    pub fn add(mut self, check: Box<dyn DiagnosticCheck>) -> Self {
        self.checks.push(check);
        self
    }

    pub fn run(self) -> bool {
        let results: Vec<(&'static str, bool)> = self
            .checks
            .iter()
            .map(|check| {
                println!("--- {} ---", check.name());
                let pass = check.run();
                println!("{} -> {}\n", check.name(), if pass { "PASS" } else { "FAIL" });
                (check.name(), pass)
            })
            .collect();

        println!("================= SUMMARY =================");
        for (name, pass) in &results {
            println!("{:<32} {}", name, if *pass { "PASS" } else { "FAIL" });
        }
        results.iter().all(|(_, pass)| *pass)
    }
}

impl Default for DiagnosticSuite {
    fn default() -> Self {
        Self::new()
    }
}
