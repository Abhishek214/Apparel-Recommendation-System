def create_model(self, pretrained_path=None):
    """Create EfficientDet model and load state_dict weights"""
    model_config = self.config['model']
    
    print("Creating EfficientDet architecture...")
    
    # Create EfficientDet-D1 architecture manually
    try:
        import timm
        # Create the model architecture without pretrained weights
        self.model = timm.create_model(
            'tf_efficientdet_d1', 
            pretrained=False,  # Don't download weights
            num_classes=model_config['num_classes']
        )
        print("✅ Created EfficientDet-D1 architecture using timm")
        
    except ImportError:
        print("timm not available, using fallback architecture")
        # Fallback to simple architecture if timm not available
        self.model = SimpleEfficientDet(
            num_classes=model_config['num_classes'],
            image_size=model_config['image_size']
        )
    
    # Now load your state_dict weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading state_dict from: {pretrained_path}")
        
        try:
            # Load the state dict
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint  # Assume it's the state_dict itself
            else:
                raise ValueError("Unexpected checkpoint format")
            
            # Get current model state dict
            model_dict = self.model.state_dict()
            
            # Filter and load compatible weights
            pretrained_dict = {}
            skipped_keys = []
            loaded_keys = []
            
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                key = k.replace('module.', '')
                
                if key in model_dict:
                    if model_dict[key].shape == v.shape:
                        pretrained_dict[key] = v
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {model_dict[key].shape} vs {v.shape})")
                else:
                    skipped_keys.append(f"{key} (not found in model)")
            
            # Update model dict and load
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
            print(f"✅ Loaded {len(loaded_keys)} layers from pretrained model")
            if len(skipped_keys) > 10:
                print(f"⚠️  Skipped {len(skipped_keys)} incompatible layers")
            
            # If we loaded some backbone weights but need to modify classifier
            if loaded_keys and model_config['num_classes'] != 90:  # COCO has 90 classes
                self.modify_classifier_for_classes(model_config['num_classes'])
                
        except Exception as e:
            print(f"❌ Error loading state_dict: {e}")
            print("Continuing with random initialization...")
    
    # Move to device
    self.model = self.model.to(self.device)
    self.model.train()
    
    # Create EMA if enabled
    if self.config['training']['model_ema']['enabled']:
        self.model_ema = ModelEMA(
            self.model,
            decay=self.config['training']['model_ema']['decay']
        )
    
    # Print model info
    total_params = sum(p.numel() for p in self.model.parameters())
    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    print(f"Model created successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def modify_classifier_for_classes(self, num_classes):
    """Modify the classifier layers for your number of classes"""
    try:
        # EfficientDet has class_net for classification
        if hasattr(self.model, 'class_net'):
            # Find the final prediction layer
            for name, module in self.model.class_net.named_modules():
                if isinstance(module, nn.Conv2d) and 'predict' in name:
                    # Modify this layer for your classes
                    old_out_channels = module.out_channels
                    new_out_channels = num_classes * 9  # 9 anchors per location
                    
                    new_conv = nn.Conv2d(
                        module.in_channels,
                        new_out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=(module.bias is not None)
                    )
                    
                    # Initialize with small random weights
                    nn.init.normal_(new_conv.weight, std=0.01)
                    if new_conv.bias is not None:
                        nn.init.constant_(new_conv.bias, 0)
                    
                    # Replace the layer
                    setattr(self.model.class_net, name.split('.')[-1], new_conv)
                    print(f"✅ Modified {name}: {old_out_channels} -> {new_out_channels} channels")
                    break
        
        print(f"✅ Classifier modified for {num_classes} classes")
        
    except Exception as e:
        print(f"⚠️  Could not modify classifier: {e}")
        print("Model will use original classification head")
