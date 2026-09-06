    use super::*;

    const LOGITS: [f32; 5] = [1.0, 4.0, 2.0, 8.0, 3.0];

    #[test]
    fn zero_temperature_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 0.0, 0, 1.0, &[], 1.0), 3);
        }
    }

    #[test]
    fn top_k_one_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 1.5, 1, 1.0, &[], 1.0), 3);
        }
    }

    #[test]
    fn top_k_restricts_candidates() {
        for _ in 0..50 {
            let t = sample_token(&LOGITS, 2.0, 2, 1.0, &[], 1.0);
            assert!(t == 3 || t == 1, "sampled outside top-2: {t}");
        }
    }

    #[test]
    fn tiny_top_p_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 1.0, 0, 0.01, &[], 1.0), 3);
        }
    }

    #[test]
    fn filters_off_can_reach_every_token() {
        let mut seen = [false; 5];
        for _ in 0..2000 {
            seen[sample_token(&LOGITS, 100.0, 0, 1.0, &[], 1.0) as usize] = true;
        }
        assert_eq!(seen, [true; 5], "some tokens never sampled: {seen:?}");
    }

    #[test]
    fn repetition_penalty_demotes_seen_token_in_greedy_mode() {
        assert_eq!(sample_token(&LOGITS, 0.0, 0, 1.0, &[], 1.0), 3);
        assert_eq!(sample_token(&LOGITS, 0.0, 0, 1.0, &[3], 100.0), 1);
    }
