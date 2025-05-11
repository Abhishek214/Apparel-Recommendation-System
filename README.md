
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


import optuna
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

def create_lora_config(trial):
    """
    Create LoRA configuration with hyperparameters suggested by Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        LoraConfig object with optimized hyperparameters
    """
    # LoRA hyperparameters to optimize
    r = trial.suggest_int("lora_r", 8, 64, log=True)  # LoRA rank
    lora_alpha = trial.suggest_int("lora_alpha", 8, 64, log=True)  # LoRA scaling factor
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.5)  # LoRA dropout
    
    # Target modules - could be optimized but often model-dependent
    # For simplicity, we'll keep the existing targets which seem reasonable
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "linear", 
                      "Conv2d", "lm_head", "fc2", "gate_proj", "down_proj", "up_proj"]
    
    # Whether to use rank-stabilized LoRA
    use_rslora = trial.suggest_categorical("use_rslora", [True, False])
    
    # Weight initialization method
    init_lora_weights = trial.suggest_categorical("init_lora_weights", 
                                                ["gaussian", "loftq"])
    
    # Configuration dictionary to return
    config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "lora_dropout": lora_dropout,
        "bias": "none",  # Could be optimized: "none", "all", "lora_only"
        "inference_mode": False,
        "use_rslora": use_rslora,
        "init_lora_weights": init_lora_weights,
        "modules_to_save": ["lm_head", "embed_tokens"],
    }
    
    # Add loftq_config if loftq initialization is selected
    if init_lora_weights == "loftq":
        # Optimize loftq-specific parameters
        loftq_bits = trial.suggest_categorical("loftq_bits", [4, 8])  # Quantization bits
        loftq_iter = trial.suggest_int("loftq_iter", 1, 3)  # Number of optimization iterations
        
        config_kwargs["loftq_config"] = {
            "loftq_bits": loftq_bits,
            "loftq_iter": loftq_iter
        }
        
    return LoraConfig(**config_kwargs)

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    Optimizes both training and LoRA hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        best_val_loss: Best validation loss achieved during training
    """
    # Training hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])
    
    # Create LoRA configuration
    lora_config = create_lora_config(trial)
    
    # Print trial info
    print(f"\n==== Trial {trial.number} ====")
    print(f"Training params: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
    print(f"LoRA params: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    print(f"Using RS-LoRA: {lora_config.use_rslora}, Init: {lora_config.init_lora_weights}")
    
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
    
    # Create a fresh model with the new LoRA config
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train for fewer epochs during optimization
    try:
        val_loss = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=peft_model, 
            processor=processor, 
            epochs=3,  # Short training for quicker trials
            lr=lr,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            patience=2,  # Short patience for quicker trials
            optuna_trial=True
        )
        
        # Optuna has a pruning mechanism to stop unpromising trials early
        trial.report(val_loss, 3)  # Report after 3 epochs
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        return val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')  # Return a bad score if trial fails

def run_hyperparameter_search():
    """
    Run hyperparameter search with Optuna.
    
    Returns:
        best_params: Dictionary with optimized hyperparameters
    """
    # Create a pruner that stops unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    
    # Create a new study object
    study = optuna.create_study(
        direction="minimize",  # We want to minimize validation loss
        pruner=pruner,
        study_name="florence2_finetune"
    )
    
    # Start the optimization with 20 trials
    n_trials = 20
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n==== Hyperparameter Optimization Results ====")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Validation Loss): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Visualize optimization results if possible
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("optuna_history.png")
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig("optuna_importance.png")
        
        print("Saved optimization plots to 'optuna_history.png' and 'optuna_importance.png'")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return best_trial.params

# Example usage:
if __name__ == "__main__":
    # Run hyperparameter search first
    best_params = run_hyperparameter_search()
    
    # Set up final training with best hyperparameters
    print("\n==== Starting final training with best parameters ====")
    
    # Extract and set the best hyperparameters
    BATCH_SIZE = best_params.get("batch_size", 32)
    LR = best_params.get("lr", 2e-5)
    WEIGHT_DECAY = best_params.get("weight_decay", 0.01)
    GRADIENT_ACCUMULATION_STEPS = best_params.get("gradient_accumulation_steps", 4)
    
    # Create the final LoRA configuration with proper handling of loftq
    lora_kwargs = {
        "r": best_params.get("lora_r", 16),
        "lora_alpha": best_params.get("lora_alpha", 32),
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2", "gate_proj", "down_proj", "up_proj"],
        "task_type": "CAUSAL_LM",
        "lora_dropout": best_params.get("lora_dropout", 0.1),
        "bias": "none",
        "inference_mode": False,
        "use_rslora": best_params.get("use_rslora", True),
        "init_lora_weights": best_params.get("init_lora_weights", "gaussian"),
        "modules_to_save": ["lm_head", "embed_tokens"],
    }
    
    # Add loftq_config if necessary
    if best_params.get("init_lora_weights") == "loftq":
        lora_kwargs["loftq_config"] = {
            "loftq_bits": best_params.get("loftq_bits", 8),
            "loftq_iter": best_params.get("loftq_iter", 1)
        }
        print(f"Using LoFTQ initialization with bits={best_params.get('loftq_bits', 8)}, iter={best_params.get('loftq_iter', 1)}")
    
    lora_config = LoraConfig(**lora_kwargs)
    
    # Create dataloaders with best batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    
    # Initialize model with best LoRA config
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Train the model with best hyperparameters
    EPOCHS = 30
    train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=peft_model, 
        processor=processor, 
        epochs=EPOCHS, 
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        patience=5  # More patience for final training
    )
