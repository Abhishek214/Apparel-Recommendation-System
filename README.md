
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

7.CNN

Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            # Track loss
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log at specified intervals
            if (batch_idx + 1) % (log_interval * gradient_accumulation_steps) == 0:
                logger.info(f"Epoch: {epoch+1}/{start_epoch+epochs}, Batch: {batch_idx+1}/{len(train_loader)}, "
                           f"Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

        # End of epoch
        avg_train_loss = train_loss / max(train_steps, 1)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Avg Training Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0

        # Progress bar for validation
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                        desc=f"Validation Epoch {epoch+1}/{start_epoch+epochs}")

        with torch.no_grad():
            for batch_idx, (inputs, answers) in val_pbar:
                # Skip empty batches
                if any(tensor.numel() == 0 for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
                    continue

                # Prepare labels
                try:
                    labels = processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).input_ids.to(DEVICE)
                except Exception as e:
                    logger.error(f"Error processing validation labels for batch {batch_idx}: {str(e)}")
                    continue

                # Forward pass
                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )

                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_steps += 1

                # Update progress bar
                val_pbar.set_postfix({'val_loss': loss.item()})

        # Calculate average validation loss
        avg_val_loss = val_loss / max(val_steps, 1)
        val_losses.append(avg_val_loss)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")

        # Run inference on some validation samples
        if visualize_samples > 0 and (epoch + 1) % save_epoch_interval == 0:
            inference_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_inferences")
            florence2_inference_results(model, processor, val_loader.dataset, 
                                       num_samples=visualize_samples, 
                                       output_dir=inference_dir)

        # Save checkpoint
        if (epoch + 1) % save_epoch_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            
            # Get model state dict, accounting for DDP
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            
            # Get model state dict, accounting for DDP
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
            
            logger.info(f"Best model saved with validation loss: {best_val_loss:.6f}")
        else:
            early_stop_counter += 1
            logger.info(f"Early stopping counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Plot training curves after each epoch
        plot_training_curves(train_losses, val_losses, 
                           save_path=os.path.join(checkpoint_dir, "training_curves.png"))

    # Final evaluation and cleanup
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Return the best model path for easy loading
    return os.path.join(checkpoint_dir, "best_model.pt")


def main():
    """Main function to orchestrate the training process"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Florence 2 fine-tuning for document layout parsing")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"],
                      help="Model size to use (base or large)")
    parser.add_argument("--train_data", type=str, required=True, 
                      help="Path to training data JSONL file")
    parser.add_argument("--train_images", type=str, required=True,
                      help="Path to training images directory")
    parser.add_argument("--val_data", type=str, required=True,
                      help="Path to validation data JSONL file")
    parser.add_argument("--val_images", type=str, required=True,
                      help="Path to validation images directory")
    parser.add_argument("--epochs", type=int, default=20,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                      help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_checkpoints",
                      help="Directory to save checkpoints")
    parser.add_argument("--tune_hyperparams", action="store_true",
                      help="Run hyperparameter tuning before training")
    parser.add_argument("--continue_training", action="store_true",
                      help="Continue from latest checkpoint")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loader workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                      help="Number of gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Set up the model path based on size
    model_path = MODEL_LARGE if args.model_size == "large" else MODEL_BASE
    logger.info(f"Using model: {model_path}")
    
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    
    # Create datasets with class-aware augmentation
    augmentation = DocumentLayoutAugmentation(rare_class_augment_factor=2.0)
    
    train_dataset = DetectionDataset(
        jsonl_file_path=args.train_data,
        image_directory_path=args.train_images,
        transform=augmentation,
        is_training=True
    )
    
    val_dataset = DetectionDataset(
        jsonl_file_path=args.val_data,
        image_directory_path=args.val_images,
        transform=None,
        is_training=False
    )
    
    # Visualize class distribution
    plot_class_distribution(train_dataset, title="Training Class Distribution")
    plot_class_distribution(val_dataset, title="Validation Class Distribution")
    
    # Configure weighted sampler for balanced training
    if USE_DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        weights = torch.DoubleTensor(train_dataset.class_weights)
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        pin_memory=True
    )
    
    # Run hyperparameter tuning if requested
    if args.tune_hyperparams:
        logger.info("Starting hyperparameter tuning")
        best_params = optuna_hyperparameter_search(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            processor=processor,
            n_trials=10
        )
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Update hyperparameters based on tuning results
        lora_r = best_params.get('lora_r', 16)
        lora_alpha = best_params.get('lora_alpha', 32)
        lora_dropout = best_params.get('lora_dropout', 0.1)
        batch_size = best_params.get('batch_size', args.batch_size)
        lr = best_params.get('lr', args.lr)
    else:
        # Default hyperparameters
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.1
        lr = args.lr
    
    # Configure LoRA
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
    
    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)
    logger.info("LoRA model created")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} of {all_params:,} "
               f"({100 * trainable_params / all_params:.2f}%)")
    
    # Setup distributed training if multiple GPUs
    if USE_DISTRIBUTED:
        # Initialize process group
        dist.init_process_group(backend="nccl")
        peft_model = DDP(peft_model, device_ids=[torch.cuda.current_device()])
        logger.info("Distributed training setup complete")
    
    # Optimizer setup
    optimizer = AdamW(
        peft_model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=1,
        eta_min=1e-7
    )
    
    # Train the model
    best_model_path = train_model(
        model=peft_model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        patience=5,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        continue_from_checkpoint=args.continue_training,
        visualize_samples=4
    )
    
    # Load best model for final evaluation
    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    if hasattr(peft_model, 'module'):
        peft_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        peft_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save final model
    output_dir = os.path.join(args.checkpoint_dir, "final_model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and processor
    model_to_save = peft_model.module if hasattr(peft_model, 'module') else peft_model
    model_to_save.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    logger.info(f"Final model saved to {output_dir}")
    
    # Clean up
    if USE_DISTRIBUTED:
        dist.destroy_process_group()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
