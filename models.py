"""
Neural network models for Moltbook dataset analysis
Includes content classification and toxicity detection models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

class MoltbookClassifier(nn.Module):
    """Base classifier for Moltbook tasks"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=9, dropout=0.3):
        super(MoltbookClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class ContentClassifier(MoltbookClassifier):
    """Content classification model (9 categories)"""
    
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        # 9 content categories from Moltbook
        categories = [
            'General', 'Technology', 'Viewpoint', 'Economics', 
            'Promotion', 'Politics', 'Social', 'Entertainment', 'Other'
        ]
        super().__init__(model_name, num_classes=len(categories), dropout=dropout)
        
        self.categories = categories
        
    def get_category_name(self, predicted_class):
        """Get category name from predicted class index"""
        return self.categories[predicted_class]

class ToxicityClassifier(MoltbookClassifier):
    """Toxicity detection model (5 levels)"""
    
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        # 5 toxicity levels from Moltbook
        super().__init__(model_name, num_classes=5, dropout=dropout)
        
        self.toxicity_levels = [0, 1, 2, 3, 4]  # 0=Safe, 4=Highly Toxic
        
    def get_toxicity_level(self, predicted_class):
        """Get toxicity level from predicted class index"""
        return self.toxicity_levels[predicted_class]
    
    def get_toxicity_description(self, predicted_class):
        """Get description of toxicity level"""
        descriptions = {
            0: "Safe - No harmful content detected",
            1: "Low - Minimal risk content",
            2: "Medium - Moderately concerning content",
            3: "High - Seriously concerning content", 
            4: "Severe - Highly toxic or dangerous content"
        }
        return descriptions.get(predicted_class, "Unknown level")

class MultiTaskMoltbookModel(nn.Module):
    """Multi-task model for both content classification and toxicity detection"""
    
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        super(MultiTaskMoltbookModel, self).__init__()
        
        self.model_name = model_name
        
        # Shared transformer backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Task-specific heads
        self.dropout = nn.Dropout(dropout)
        
        # Content classification head (9 categories)
        self.content_classifier = nn.Linear(self.config.hidden_size, 9)
        
        # Toxicity detection head (5 levels)
        self.toxicity_classifier = nn.Linear(self.config.hidden_size, 5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.content_classifier.weight)
        nn.init.zeros_(self.content_classifier.bias)
        nn.init.xavier_uniform_(self.toxicity_classifier.weight)
        nn.init.zeros_(self.toxicity_classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass for both tasks"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for both tasks
        content_logits = self.content_classifier(pooled_output)
        toxicity_logits = self.toxicity_classifier(pooled_output)
        
        return {
            'content_logits': content_logits,
            'toxicity_logits': toxicity_logits
        }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-smooth_targets * log_probs, dim=-1))

# Utility functions
def get_model(task='content', model_name='bert-base-uncased', dropout=0.3):
    """Get appropriate model based on task"""
    
    if task == 'content':
        return ContentClassifier(model_name, dropout)
    elif task == 'toxicity':
        return ToxicityClassifier(model_name, dropout)
    elif task == 'multi':
        return MultiTaskMoltbookModel(model_name, dropout)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'content', 'toxicity', or 'multi'")

def get_loss_function(loss_type='crossentropy', **kwargs):
    """Get appropriate loss function"""
    
    if loss_type == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Example usage
if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Content classifier
    content_model = ContentClassifier()
    content_model.to(device)
    
    # Toxicity classifier
    toxicity_model = ToxicityClassifier()
    toxicity_model.to(device)
    
    # Multi-task model
    multi_model = MultiTaskMoltbookModel()
    multi_model.to(device)
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    # Test content classifier
    content_logits = content_model(input_ids, attention_mask)
    print(f"Content logits shape: {content_logits.shape}")
    
    # Test toxicity classifier
    toxicity_logits = toxicity_model(input_ids, attention_mask)
    print(f"Toxicity logits shape: {toxicity_logits.shape}")
    
    # Test multi-task model
    multi_outputs = multi_model(input_ids, attention_mask)
    print(f"Multi-task content logits shape: {multi_outputs['content_logits'].shape}")
    print(f"Multi-task toxicity logits shape: {multi_outputs['toxicity_logits'].shape}")
    
    print("Models initialized successfully!")
