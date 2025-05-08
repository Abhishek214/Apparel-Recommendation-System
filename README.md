
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

7.CNN
def run_training(
    model_size="base",
    train_data=None,
    train_images=None,
    val_data=None,
    val_images=None,
    epochs=20,
    batch_size=16,
    lr=1e-5,
    checkpoint_dir="./model_checkpoints",
    tune_hyperparams=False,
    continue_training=False,
    num_workers=4,
    gradient_accumulation_steps=2
):
    """Function to orchestrate the training process with given parameters"""

    # Set up the model path based on size
    model_path = MODEL_LARGE if model_size == "large" else MODEL_BASE
    logger.info(f"Using model: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)

    augmentation = DocumentLayoutAugmentation(rare_class_augment_factor=2.0)

    train_dataset = DetectionDataset(
        jsonl_file_path=train_data,
        image_directory_path=train_images,
        transform=augmentation,
        is_training=True
    )

    val_dataset = DetectionDataset(
        jsonl_file_path=val_data,
        image_directory_path=val_images,
        transform=None,
        is_training=False
    )

    plot_class_distribution(train_dataset, title="Training Class Distribution")
    plot_class_distribution(val_dataset, title="Validation Class Distribution")

    if USE_DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        weights = torch.DoubleTensor(train_dataset.class_weights)
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        pin_memory=True
    )

    if tune_hyperparams:
        logger.info("Starting hyperparameter tuning")
        best_params = optuna_hyperparameter_search(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            processor=processor,
            n_trials=10
        )
        logger.info(f"Best hyperparameters: {best_params}")
        lora_r = best_params.get('lora_r', 16)
        lora_alpha = best_params.get('lora_alpha', 32)
        lora_dropout = best_params.get('lora_dropout', 0.1)
        batch_size = best_params.get('batch_size', batch_size)
        lr = best_params.get('lr', lr)
    else:
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.1

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "lm_head"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
    )

    peft_model = get_peft_model(model, lora_config)
    logger.info("LoRA model created")

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} of {all_params:,} "
                f"({100 * trainable_params / all_params:.2f}%)")

    if USE_DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        peft_model = DDP(peft_model, device_ids=[torch.cuda.current_device()])
        logger.info("Distributed training setup complete")

    optimizer = AdamW(
        peft_model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    num_training_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    best_model_path = train_model(
        model=peft_model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        patience=5,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_dir=checkpoint_dir,
        continue_from_checkpoint=continue_training,
        visualize_samples=4
    )

    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)

    if hasattr(peft_model, 'module'):
        peft_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        peft_model.load_state_dict(checkpoint['model_state_dict'])

    output_dir = os.path.join(checkpoint_dir, "final_model")
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = peft_model.module if hasattr(peft_model, 'module') else peft_model
    model_to_save.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info(f"Final model saved to {output_dir}")

    if USE_DISTRIBUTED:
        dist.destroy_process_group()

    logger.info("Training completed successfully!")
