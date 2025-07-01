                   help='Source directory containing images (optional - for copying images to split directories)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--min_val_samples', type=int, default=1,
                       help='Minimum samples per category in validation set (default: 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.1 <= args.train_ratio <= 0.9):
        print("Error: train_ratio must be between 0.1 and 0.9")
        return
    
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    if args.source_images and not Path(args.source_images).exists():
        print(f"Error: Source images directory does not exist: {args.source_images}")
        return
    
    # Create splitter and split dataset
    splitter = COCODatasetSplitter(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )
    
    splitter.split_dataset(source_image_dir=args.source_images)


if __name__ == "__main__":
    import os
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("COCO Dataset Splitter")
        print("====================")
        print("\nExample usage:")
        print("python split_coco_dataset.py --input dataset.json --output_dir ./data --train_ratio 0.8")
        print("\nWith image copying:")
        print("python split_coco_dataset.py --input dataset.json --output_dir ./data --train_ratio 0.8 --source_images ./images")
        print("\nArguments:")
        print("  --input: Path to COCO JSON file")
        print("  --output_dir: Directory to create train/val splits")
        print("  --train_ratio: Fraction for training (0.1-0.9, default: 0.8)")
        print("  --source_images: Source image directory (optional)")
        print("  --random_seed: Random seed for reproducibility (default: 42)")
    else:
        main()
