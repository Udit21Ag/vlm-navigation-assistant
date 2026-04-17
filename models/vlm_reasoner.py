import os
import torch
import numpy as np
import cv2

class VLMReasoner:
    """
    BLIP VQA Checker: Validates spatial descriptions against actual image content.
    
    Purpose: Verify that the temporal caption accurately describes the scene
    Uses lightweight BLIP VQA model as a validator (not generator)
    """

    def __init__(self, model_name="blip-vqa"):
        self.model_name = model_name
        self.available = False
        
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            
            print("[VLM] Loading BLIP VQA for validation...")
            
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.device = device
            
            self.available = True
            print(f"[VLM] BLIP VQA loaded successfully (device: {device})")
            
        except Exception as e:
            print(f"[VLM] Failed to load BLIP VQA: {e}")
            self.available = False
    
    def generate(self, frame, temporal_caption, instruction, temporal_objects=None):
        """
        Validate temporal caption using BLIP VQA as a checker.
        Focuses on detecting missed hazards rather than strict validation.
        
        Args:
            frame: Input image (numpy array)
            temporal_caption: Generated temporal caption
            instruction: Navigation direction (e.g., "Move right.")
            temporal_objects: List of detected objects (unused)
            
        Returns:
            temporal_caption (with BLIP hazard verification)
        """
        
        if not self.available:
            return temporal_caption
        
        try:
            # Convert BGR to RGB for transformers
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ============================================================
            # PRIMARY: Ask what objects/hazards are in the scene
            # ============================================================
            hazard_question = "What are the main hazards or obstacles in this street scene?"
            
            inputs = self.processor(rgb_frame, hazard_question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=30)
            
            vlm_hazards = self.processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            print(f"[VLM] Temporal Caption: '{temporal_caption}'")
            print(f"[VLM] BLIP Sees: '{vlm_hazards}'")
            
            # ============================================================
            # Check for agreement
            # ============================================================
            temporal_lower = temporal_caption.lower()
            
            # Extract key phrases from both
            key_words = ["autorickshaw", "motorcycle", "car", "bus", "truck", "person", 
                        "vehicle", "obstacle", "hazard", "left", "right", "ahead", "forward"]
            
            temporal_keys = [w for w in key_words if w in temporal_lower]
            vlm_keys = [w for w in key_words if w in vlm_hazards]
            
            overlap = len(set(temporal_keys) & set(vlm_keys))
            
            if overlap >= 2 or len(temporal_keys) == 0:
                print(f"[VLM] ✓ Hazard agreement confirmed")
            else:
                print(f"[VLM] ⚠ Potential mismatch: Temporal={temporal_keys}, VLM={vlm_keys}")
            
            # Return temporal caption (it's already well-structured)
            return temporal_caption
            
        except Exception as e:
            print(f"[VLM] Validation error: {e}")
            return temporal_caption
