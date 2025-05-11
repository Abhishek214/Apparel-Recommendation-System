
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


# Create a new function before train_model
def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    # Recreate dataloaders with new batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    
    # Re-initialize model each trial to start fresh
    peft_model = get_peft_model(model, config)
    
    # Train for fewer epochs during optimization
    val_loss = train_model(train_loader, val_loader, peft_model, processor, 
                     epochs=5, lr=lr, weight_decay=weight_decay, optuna_trial=True)
    
    return val_loss  # Return best validation loss





# Add code to run Optuna before the final training
# Add after model definition but before final training code
def run_hyperparameter_search():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

# Uncomment to run hyperparameter search
# best_params = run_hyperparameter_search()
# BATCH_SIZE = best_params["batch_size"]
# LR = best_params["lr"]
# WEIGHT_DECAY = best_params["weight_decay"]





# Modify the train_transform (around line 156) to better handle document-specific needs
train_transform = A.Compose([
    A.OneOf([  # More randomness while preserving readability
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.8),
    ], p=0.7),
    A.OneOf([  # Simulate scanning artifacts
        A.GaussianBlur(blur_limit=(1, 3), p=0.8),
        A.MotionBlur(blur_limit=(3, 5), p=0.8),
    ], p=0.5),
    A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    A.Perspective(scale=(0.01, 0.03), p=0.3),
    A.Rotate(limit=2, p=0.7),  # Small rotations only
    A.ToGray(p=0.2),  # Sometimes convert to grayscale
    ToTensorV2()  # Add this to convert to tensor directly
])




# Update the __getitem__ method in DetectionDataset to properly handle transforms
# Find the __getitem__ method in DetectionDataset class (around line 119)
def __getitem__(self, idx):
    image, data = self.dataset[idx]
    prefix = data['prefix']
    suffix = data['suffix']
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to numpy for albumentations
    image_np = np.array(image)
    
    # Apply transforms if available
    if self.transform and self.is_training:
        transformed = self.transform(image=image_np)
        # With ToTensorV2() in the transform pipeline, this returns a tensor
        image_tensor = transformed['image']
        return prefix, suffix, image_tensor
    else:
        # Convert to tensor for validation
        image_tensor = transforms.ToTensor()(image)
        return prefix, suffix, image_tensor





# Update training loop to use mixed precision where gradient computation happens
# Inside the train loop around line 310
if scaler is not None:
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
    
    # Scale the loss and backpropagate
    scaler.scale(loss).backward()
    
    # Skip NaN gradients check
    if not check_for_nan_gradients(model):
        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        scaler.update()  # Update scaler even if we skip step
    
    optimizer.zero_grad()
else:
    # Original non-mixed precision code



