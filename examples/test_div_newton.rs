use spirix::ScalarF4E4;

fn main() {
    println!("Testing Newton-Raphson Division:\n");

    let test_cases = vec![
        (7, 3),      // 7/3 = 2.333...
        (10, 2),     // 10/2 = 5
        (1, 1),      // 1/1 = 1        (15, 4),     // 15/4 = 3.75
        (100, 7),    // 100/7 = 14.28...
        (2, 5),      // 2/5 = 0.4
    ];

    for (a, b) in test_cases {
        let x = ScalarF4E4::from(a);
        let y = ScalarF4E4::from(b);
        let result = x / y;
        let expected = a as f64 / b as f64;
        let actual_f64: f64 = result.into();

        println!("{:3} / {:3} = {:.6} (expected: {:.6}, error: {:.6})",
                 a, b, actual_f64, expected, (actual_f64 - expected).abs());
    }
}
