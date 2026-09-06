    use super::eval_windows;

    #[test]
    fn eval_windows_are_fixed_shifted_and_batch_aligned() {
        let tokens: Vec<u32> = (0..41).collect();
        let set = eval_windows(&tokens, 4, 2, 32).unwrap();

        // 40 usable tokens / 4 = 10 windows, floored to a multiple of 2.
        assert_eq!(set.windows, 10);
        assert_eq!(set.inputs.len(), 40);
        // Non-overlapping consecutive windows, targets shifted by one.
        assert_eq!(&set.inputs[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&set.targets[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(set.inputs[36..40], [36, 37, 38, 39]);
        assert_eq!(set.targets[39], 40);

        // Batch alignment floors odd window counts.
        let set = eval_windows(&(0..29).collect::<Vec<u32>>(), 4, 2, 32).unwrap();
        assert_eq!(set.windows, 6);

        // Too small for a single batch -> None, not a panic.
        assert!(eval_windows(&(0..8).collect::<Vec<u32>>(), 4, 2, 32).is_none());
        assert!(eval_windows(&[], 4, 2, 32).is_none());

        // max_eval_windows caps the count even when more tokens are available.
        let set = eval_windows(&(0..41).collect::<Vec<u32>>(), 4, 2, 4).unwrap();
        assert_eq!(set.windows, 4);
    }
