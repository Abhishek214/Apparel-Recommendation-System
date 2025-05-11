
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
import torch
import os
import shutil

def objective(trial):
    """
    Optuna objective function with improved error handling.
    """
    # Use smaller parameter ranges initially to find a stable configuration
    lr = trial.suggest_float("lr", 5e-6, 2e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])  # Smaller batch sizes
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)  # Higher weight decay range
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16])  # Larger accumulation
    
    # LoRA hyperparameters - more conservative ranges
    r = trial.suggest_int("lora_r", 8, 32)
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64)  # Higher alpha for stability
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)  # Higher dropout range
    
    # Other LoRA parameters
    use_rslora = trial.suggest_categorical("use_rslora", [True])  # Always use RS-LoRA for stability
    lora_weights = trial.suggest_categorical("lora_weights", ["gaussian"])  # Stick with gaussian init
    
    # Print trial info
    print(f"\n==== Trial {trial.number} ====")
    print(f"Training params: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
    print(f"LoRA params: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Using RS-LoRA: {use_rslora}, Init: {lora_weights}")
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", 
                        "Conv2d", "lm_head", "fc2", "gate_proj", "down_proj", "up_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias="none",
        inference_mode=False,
        use_rslora=use_rslora,
        init_lora_weights=lora_weights,
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    try:
        # Recreate dataloaders with new batch size and the FIXED collate_fn
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            collate_fn=collate_fn,  # Make sure this is the fixed version
            num_workers=0,  # Use 0 workers for debugging
            pin_memory=True,
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn,  # Make sure this is the fixed version
            num_workers=0,  # Use 0 workers for debugging
            pin_memory=True,
            persistent_workers=False
        )
        
        # Initialize a fresh base model for each trial
        print("Initializing base model...")
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        
        # Apply LoRA configuration
        print("Applying LoRA configuration...")
        peft_model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create a trial checkpoint directory
        trial_dir = f"./optuna_checkpoints/trial_{trial.number}"
        os.makedirs(trial_dir, exist_ok=True)
        
        # Train for only 1-2 epochs initially to quickly find promising configurations
        val_loss = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=peft_model, 
            processor=processor, 
            epochs=2,  # Very short training for initial trials
            lr=lr,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            patience=3,  # Higher patience for more stable evaluation
            optuna_trial=True,
            save_dir=trial_dir
        )
        
        # Save the configuration for this trial
        with open(f"{trial_dir}/config.txt", "w") as f:
            f.write(f"Trial {trial.number}\n")
            f.write(f"Validation Loss: {val_loss}\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # Clean up to prevent CUDA OOM errors between trials
        del peft_model
        del base_model
        torch.cuda.empty_cache()
        
        return val_loss
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Clean up in case of failure
        try:
            del peft_model
        except:
            pass
        try:
            del base_model
        except:
            pass
        torch.cuda.empty_cache()
        
        # Return a high loss value to indicate failure
        return float('inf')

def run_hyperparameter_search(n_trials=10):
    """
    Run hyperparameter search with Optuna.
    More conservative approach with fewer trials initially.
    """
    # Create output directory
    os.makedirs("./optuna_checkpoints", exist_ok=True)
    
    # Create a pruner that stops unpromising trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    
    # Create a new study
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="florence2_finetune_robust"
    )
    
    # Start with fewer trials
    print(f"Starting conservative hyperparameter search with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    
    # Print results
    print("\n==== Hyperparameter Optimization Results ====")
    print("Best trial:")
    
    if len(study.trials) > 0 and hasattr(study, 'best_trial'):
        best_trial = study.best_trial
        print(f"  Value (Best Validation Loss): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best trial parameters to a file
        with open("./optuna_best_params.txt", "w") as f:
            f.write(f"Best Trial: {best_trial.number}\n")
            f.write(f"Validation Loss: {best_trial.value}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        
        return best_trial.params
    else:
        print("  No successful trials completed.")
        # Return default parameters as fallback
        return {
            "lr": 1e-5,
            "batch_size": 16,
            "weight_decay": 0.05,
            "gradient_accumulation_steps": 8,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "use_rslora": True,
            "lora_weights": "gaussian"
        }

# Modified train_model function to support trial-specific save directories
def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-5,
               weight_decay=0.01, gradient_accumulation_steps=4, patience=5, 
               optuna_trial=False, save_dir="./model_checkpoints"):
    """
    Modified train_model function with trial-specific save directory support.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Rest of function remains the same as in the previous artifact
    # ...
    
    # Return best validation loss
    return best_val_loss

