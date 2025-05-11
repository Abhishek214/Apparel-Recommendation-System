
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

# SOLUTION 1: Disable Mixed Precision Training
def train_model_no_amp(train_loader, val_loader, model, processor, epochs=10, lr=1e-5,
                      weight_decay=0.01, gradient_accumulation_steps=4, patience=5, 
                      optuna_trial=False):
    """
    Training function with mixed precision disabled to avoid FP16 issues.
    """
    os.makedirs("./model_checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        eps=1e-6
    )
    
    num_training_steps = epochs * len(train_loader) // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # NO SCALER - Train in FP32
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        accumulated_batches = 0
        
        for batch_idx, (inputs, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            try:
                # Move inputs to device
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)
                
                # Tokenize answers
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).input_ids.to(DEVICE)
                
                # Forward pass WITHOUT autocast
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                    attention_mask=attention_mask
                )
                
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                train_loss += loss.item() * gradient_accumulation_steps
                accumulated_batches += 1
                
                # Optimization step
                if accumulated_batches % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    batch_count += 1
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue
        
        # Validation phase (similar, without mixed precision)
        # ... rest of the training loop ...
    
    return best_val_loss

# SOLUTION 2: Properly Configure Mixed Precision Training
def train_model_fixed_amp(train_loader, val_loader, model, processor, epochs=10, lr=1e-5,
                         weight_decay=0.01, gradient_accumulation_steps=4, patience=5, 
                         optuna_trial=False):
    """
    Training function with properly configured mixed precision.
    """
    os.makedirs("./model_checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        eps=1e-6
    )
    
    num_training_steps = epochs * len(train_loader) // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Initialize scaler with proper settings
    scaler = GradScaler(
        init_scale=128,  # Lower initial scale
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100,  # More frequent adjustments
        enabled=torch.cuda.is_available()  # Only enable on CUDA
    )
    
    # Track gradient scaling issues
    scale_factor_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        accumulated_batches = 0
        skip_count = 0
        
        for batch_idx, (inputs, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            try:
                # Move inputs to device
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)
                
                # Tokenize answers
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).input_ids.to(DEVICE)
                
                # Forward pass with autocast
                with autocast(dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask=attention_mask
                    )
                    
                    loss = outputs.loss
                    if loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                train_loss += loss.item() * gradient_accumulation_steps
                accumulated_batches += 1
                
                # Optimization step
                if accumulated_batches % gradient_accumulation_steps == 0:
                    # Record current scale
                    current_scale = scaler.get_scale()
                    scale_factor_history.append(current_scale)
                    
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                    # Check for inf/nan gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"Skipping batch {batch_idx} due to inf/nan gradients")
                        skip_count += 1
                        optimizer.zero_grad()
                        scaler.update()  # Will reduce scale factor
                        continue
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    batch_count += 1
                    
                    # Log scaling info periodically
                    if batch_count % 10 == 0:
                        new_scale = scaler.get_scale()
                        print(f"Batch {batch_count}: Scale factor: {new_scale:.1f}, "
                              f"Grad norm: {grad_norm:.3f}, Skip rate: {skip_count/batch_count:.1%}")
                    
            except RuntimeError as e:
                if "unscale" in str(e):
                    print(f"Gradient scaling error in batch {batch_idx}: {e}")
                    # Reset optimizer state
                    optimizer.zero_grad()
                    # Reduce scale factor
                    new_scale = scaler.get_scale() * 0.5
                    scaler.update(new_scale)
                    print(f"Reduced scale factor to: {new_scale}")
                else:
                    raise e
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue
        
        # Log epoch statistics
        print(f"Epoch {epoch+1} complete. Skipped batches: {skip_count}")
        if scale_factor_history:
            print(f"Scale factor range: {min(scale_factor_history):.1f} - {max(scale_factor_history):.1f}")
        
        # Validation phase (without mixed precision for stability)
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, answers) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")):
                # Regular FP32 validation
                # ... validation code ...
        
        # Early stopping logic
        # ... rest of the training loop ...
    
    return best_val_loss

# SOLUTION 3: Use BF16 Instead of FP16 (if available)
def train_model_bf16(train_loader, val_loader, model, processor, epochs=10, lr=1e-5,
                    weight_decay=0.01, gradient_accumulation_steps=4, patience=5, 
                    optuna_trial=False):
    """
    Training with BF16 which is more stable than FP16.
    """
    # Check if BF16 is available
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("Using BF16 mixed precision training")
        dtype = torch.bfloat16
    else:
        print("BF16 not available, using FP32")
        dtype = torch.float32
    
    # ... rest of training code with autocast(dtype=dtype) ...

# SOLUTION 4: Modified Optuna Objective Without Mixed Precision
def stable_objective(trial):
    """
    Optuna objective that disables mixed precision for stability.
    """
    # Conservative parameters
    lr = trial.suggest_float("lr", 1e-7, 1e-6, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32])
    
    # LoRA parameters
    lora_r = trial.suggest_int("lora_r", 4, 8)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
    )
    
    try:
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        # Initialize model in FP32
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use FP32 for stability
        ).to(DEVICE)
        
        # Apply LoRA
        peft_model = get_peft_model(base_model, lora_config)
        
        # Train without mixed precision
        val_loss = train_model_no_amp(  # Use the version without AMP
            train_loader=train_loader,
            val_loader=val_loader,
            model=peft_model,
            processor=processor,
            epochs=2,
            lr=lr,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            patience=2,
            optuna_trial=True
        )
        
        # Cleanup
        del peft_model
        del base_model
        torch.cuda.empty_cache()
        
        return val_loss
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# RECOMMENDATION: Quick fix for your current issue
# Replace scaler-related code in your train_model function with:
"""
# Option 1: Disable mixed precision completely
# Remove all scaler code and autocast blocks

# Option 2: Use more conservative scaler settings
scaler = GradScaler(
    init_scale=16,  # Much lower initial scale
    growth_factor=1.5,  # Slower growth
    backoff_factor=0.5,
    growth_interval=500,  # Less frequent adjustments
    enabled=True
)

# Option 3: Switch to BF16 if available
if torch.cuda.is_bf16_supported():
    autocast_dtype = torch.bfloat16
else:
    # Disable mixed precision
    use_mixed_precision = False
"""
