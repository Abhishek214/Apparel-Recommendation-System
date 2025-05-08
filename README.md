
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

7.CNN
""Plot class distribution from dataset"""
class_counts = dataset.dataset.class_counts

plt.figure(figsize=(12, 6))
bars = plt.bar(class_counts.keys(), class_counts.values())

# Add count labels on top of bars
for bar in bars:
height = bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
f'{height:.0f}', ha='center', va='bottom')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title(title)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{title.lower().replace(' ', '_')}.png")
plt.close()

def florence2_inference_results(model, processor, dataset, num_samples=3,
output_dir="inference_results"):
"""Run inference on Florence 2 model and visualize results"""
# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Ensure model is in eval mode
model.eval()

# Get the actual model if wrapped in DataParallel or DDP
if hasattr(model, 'module'):
model_to_use = model.module
else:
model_to_use = model

# Get random sample indices
indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

results = []
for i in indices:
prefix, suffix, image = dataset[i]

# Convert PIL image to tensor if needed
if not isinstance(image, torch.Tensor):
image_tensor = transforms.ToTensor()(image)
else:
image_tensor = image

# Ensure 3 channels
if image_tensor.shape[0] == 1:
image_tensor = image_tensor.repeat(3, 1, 1)

with torch.no_grad():
# Prepare inputs
inputs = processor(text=prefix, images=image_tensor.unsqueeze(0),
return_tensors="pt").to(model_to_use.device)

# Generate prediction
generated_ids = model_to_use.generate(
input_ids=inputs["input_ids"],
pixel_values=inputs["pixel_values"],
max_new_tokens=1024,
early_stopping=False,
do_sample=False,
num_beams=3,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Parse the generated output
try:
parsed_answer = processor.post_process_generation(
generated_text,
task='<OD>',
image_size=(image.width, image.height))

# Access bounding boxes and labels
od_results = parsed_answer.get('<OD>', {})
bboxes = od_results.get('bboxes', [])
labels = od_results.get('labels', [])

# Save visualization
save_path = os.path.join(output_dir, f"inference_sample_{i}.png")
plot_bboxes_on_image(image, bboxes, labels, save_path)

results.append({
'sample_idx': i,
'prefix': prefix,
'bboxes': bboxes,
'labels': labels,
'image_path': save_path
})

logger.info(f"Processed inference sample {i} with {len(bboxes)} detections")
except Exception as e:
logger.error(f"Error processing inference for sample {i}: {str(e)}")

return results

# Main training function with multi-GPU support
def train_model(model, processor, train_loader, val_loader,
optimizer, scheduler=None, epochs=20,
gradient_accumulation_steps=2,
patience=5, log_interval=10,
checkpoint_dir="./model_checkpoints",
save_epoch_interval=1,
visualize_samples=4,
continue_from_checkpoint=True):
"""Training function with multi-GPU support, mixed precision, and early stopping"""

# Create checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize mixed precision training if available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Initialize tracking variables
best_val_loss = float('inf')
early_stop_counter = 0
start_epoch = 0
train_losses = []
val_losses = []

# Load checkpoint if available and requested
if continue_from_checkpoint:
# Try loading best model checkpoint
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
if os.path.exists(best_model_path):
logger.info(f"Loading checkpoint from {best_model_path}")
try:
checkpoint = torch.load(best_model_path, map_location=DEVICE)

# Get the right model (unwrap DataParallel/DDP if needed)
model_to_load = model.module if hasattr(model, 'module') else model
model_to_load.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_val_loss = checkpoint['val_loss']

# Load training history if available
if 'train_losses' in checkpoint:
train_losses = checkpoint['train_losses']
if 'val_losses' in checkpoint:
val_losses = checkpoint['val_losses']

logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.6f}")
except Exception as e:
logger.error(f"Error loading checkpoint: {str(e)}")
logger.info("Starting training from scratch")
else:
logger.info("No checkpoint found, starting training from scratch")

# Main training loop
for epoch in range(start_epoch, start_epoch + epochs):
epoch_start_time = time.time()

# Training phase
model.train()
train_loss = 0
train_steps = 0

# Progress bar for training
pbar = tqdm(enumerate(train_loader), total=len(train_loader),
desc=f"Epoch {epoch+1}/{start_epoch+epochs}")

# Batch processing
for batch_idx, (inputs, answers) in pbar:
# Skip empty batches
if any(tensor.numel() == 0 for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
logger.warning(f"Skipping empty batch {batch_idx}")
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
logger.error(f"Error processing labels for batch {batch_idx}: {str(e)}")
continue

# Forward pass with mixed precision
try:
with torch.cuda.amp.autocast() if scaler else torch.no_grad():
outputs = model(
input_ids=inputs["input_ids"],
pixel_values=inputs["pixel_values"],
labels=labels
)

loss = outputs.loss
if loss.dim() > 0:
loss = loss.mean()

# Scale loss by gradient accumulation steps
loss = loss / gradient_accumulation_steps
except RuntimeError as e:
if "CUDA out of memory" in str(e):
logger.error(f"CUDA OOM in batch {batch_idx}, skipping")
# Try to free memory
torch.cuda.empty_cache()
continue
else:
logger.error(f"Error in forward pass for batch {batch_idx}: {str(e)}")
continue

# Check if loss is valid
if torch.isnan(loss) or torch.isinf(loss):
logger.warning(f"Invalid loss value: {loss.item()} in batch {batch_idx}, skipping")
continue

# Backward pass with mixed precision
if scaler:
scaler.scale(loss).backward()
else:
loss.backward()

# Check for NaN gradients
has_nan_grad = False
for name, param in model.named_parameters():
if param.grad is not None and torch.isnan(param.grad).any():
logger.warning(f"NaN gradient detected in {name}")
has_nan_grad = True
break

if has_nan_grad:
logger.warning("NaN gradients detected, skipping optimization step")
optimizer.zero_grad()
continue

# Update weights if gradient accumulation steps reached
if (batch_idx + 1) % gradient_accumulation_steps == 0:
# Clip gradients
if scaler:
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Perform optimization step
if scaler:

