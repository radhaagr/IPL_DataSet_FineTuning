#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install tf-keras


# In[ ]:


#pip install datasets peft transformers accelerate bitsandbytes


# In[1]:


import os
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import ( AutoModelForCausalLM, 
                           AutoTokenizer, 
                           TrainingArguments, 
                           Trainer,
                            DataCollatorForLanguageModeling
                          )
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from datetime import datetime


# In[2]:


class IPLJSONDatasetProcessor:
    """Process IPL dataset from JSON files across multiple directories"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.all_prompts = []
        
    def load_batting_stats(self) -> List[Dict]:
        """Load batting statistics from JSON files"""
        batting_dir = self.base_path / "batting_stats"
        prompts = []
        
        if not batting_dir.exists():
            print(f"Warning: {batting_dir} not found")
            return prompts
        
        for json_file in batting_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle the nested structure from your sample
                if isinstance(data, dict) and 'response' in data:
                    stats_list = data['response'].get('stats', [])
                elif isinstance(data, dict) and 'stats' in data:
                    stats_list = data['stats']
                elif isinstance(data, list):
                    stats_list = data
                else:
                    stats_list = [data]
                
                for stat in stats_list:
                    prompts.extend(self._create_batting_prompts(stat))
                
                print(f"Loaded: {json_file.name} - {len(stats_list)} records")
            except Exception as e:    
                print(f"Error loading {json_file}: {e}")
        
        return prompts
    
    def _create_batting_prompts(self, stat: Dict) -> List[Dict]:
        """Create training prompts from batting statistics"""
        prompts = []
        
        # Extract player and team info
        player_info = stat.get('player', {})
        team_info = stat.get('team', {})
        
        player_name = player_info.get('title', 'Unknown Player')
        team_name = team_info.get('title', 'Unknown Team')
        
        # Extract key statistics
        matches = stat.get('matches', 0)
        runs = stat.get('runs', 0)
        average = stat.get('average', 'N/A')
        strike_rate = stat.get('strike', 'N/A')
        highest_score = stat.get('highest', 0)
        centuries = stat.get('run100', 0)
        fifties = stat.get('run50', 0)
        sixes = stat.get('run6', 0)
        fours = stat.get('run4', 0)
        
        # Prompt 1: Player statistics query
        prompts.append({
            "instruction": f"What are the batting statistics for {player_name}?",
            "input": f"Team: {team_name}, Format: T20",
            "output": f"{player_name} played {matches} matches for {team_name}, scoring {runs} runs at an average of {average} with a strike rate of {strike_rate}. Highest score: {highest_score}. The player hit {centuries} centuries, {fifties} fifties, {sixes} sixes, and {fours} fours."
        })
        
        # Prompt 2: Performance analysis
        performance_desc = self._analyze_batting_performance(stat)
        prompts.append({
            "instruction": f"Analyze the batting performance of {player_name}",
            "input": f"Statistics: {runs} runs in {matches} matches",
            "output": performance_desc
        })
        
        # Prompt 3: Strike rate query
        if strike_rate != 'N/A':
            prompts.append({
                "instruction": f"What is {player_name}'s strike rate?",
                "input": f"Player: {player_name}, Team: {team_name}",
                "output": f"{player_name} has a strike rate of {strike_rate} in T20 cricket for {team_name}."
            })
        
        # Prompt 4: Comparison/ranking context
        if runs > 300:
            prompts.append({
                "instruction": "Who are the top run scorers in IPL?",
                "input": f"Consider players with over 300 runs",
                "output": f"{player_name} from {team_name} scored {runs} runs with an average of {average}, making them one of the top performers."
            })
        
        return prompts
    
    def _analyze_batting_performance(self, stat: Dict) -> str:
        """Generate analytical narrative for batting performance"""
        player_name = stat.get('player', {}).get('title', 'The player')
        runs = stat.get('runs', 0)
        average = stat.get('average', 'N/A')
        strike_rate = stat.get('strike', 'N/A')
        matches = stat.get('matches', 0)
        
        analysis = f"{player_name} demonstrated "
        
        # Analyze average
        try:
            avg_val = float(average)
            if avg_val > 40:
                analysis += "excellent consistency with a strong average, "
            elif avg_val > 30:
                analysis += "good consistency, "
            else:
                analysis += "moderate performance, "
        except:
            analysis += "notable performance, "
        
        # Analyze strike rate
        try:
            sr_val = float(strike_rate)
            if sr_val > 150:
                analysis += "playing with explosive aggression. "
            elif sr_val > 130:
                analysis += "maintaining a healthy strike rate. "
            else:
                analysis += "with a measured approach. "
        except:
            pass
        
        analysis += f"Across {matches} matches, they accumulated {runs} runs, "
        
        # Add context about centuries/fifties
        centuries = stat.get('run100', 0)
        fifties = stat.get('run50', 0)
        if centuries > 0:
            analysis += f"including {centuries} centuries, "
        if fifties > 0:
            analysis += f"with {fifties} half-centuries. "
        
        return analysis
    
    def load_bowling_stats(self) -> List[Dict]:
        """Load bowling statistics from JSON files"""
        bowling_dir = self.base_path / "bowling_stats"
        prompts = []
        
        if not bowling_dir.exists():
            print(f"Warning: {bowling_dir} not found")
            return prompts
        
        for json_file in bowling_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle nested structure
                if isinstance(data, dict) and 'response' in data:
                    stats_list = data['response'].get('stats', [])
                elif isinstance(data, dict) and 'stats' in data:
                    stats_list = data['stats']
                elif isinstance(data, list):
                    stats_list = data
                else:
                    stats_list = [data]
                
                for stat in stats_list:
                    prompts.extend(self._create_bowling_prompts(stat))
                
                print(f"Loaded: {json_file.name} - {len(stats_list)} records")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return prompts
    
    def _create_bowling_prompts(self, stat: Dict) -> List[Dict]:
        """Create training prompts from bowling statistics"""
        prompts = []
        
        player_info = stat.get('player', {})
        team_info = stat.get('team', {})
        
        player_name = player_info.get('title', 'Unknown Player')
        team_name = team_info.get('title', 'Unknown Team')
        
        matches = stat.get('matches', 0)
        wickets = stat.get('wickets', 0)
        economy = stat.get('economy', 'N/A')
        average = stat.get('average', 'N/A')
        
        # Bowling stats prompt
        prompts.append({
            "instruction": f"What are the bowling statistics for {player_name}?",
            "input": f"Team: {team_name}, Format: T20",
            "output": f"{player_name} took {wickets} wickets in {matches} matches for {team_name} with an economy of {economy} and bowling average of {average}."
        })
        
        return prompts
    
    def load_match_info(self) -> List[Dict]:
        """Load match information from JSON files"""
        match_dir = self.base_path / "match_info"
        prompts = []
        
        if not match_dir.exists():
            print(f"Warning: {match_dir} not found")
            return prompts
        
        for json_file in match_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts.extend(self._create_match_prompts(data))
                
                print(f"Loaded: {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return prompts
    
    def _create_match_prompts(self, match_data: Dict) -> List[Dict]:
        """Create prompts from match information"""
        prompts = []
        
        # Extract match details based on common IPL JSON structures
        if isinstance(match_data, dict):
            teams = match_data.get('teams', [])
            venue = match_data.get('venue', {}).get('name', 'Unknown Venue')
            match_date = match_data.get('date_start', 'Unknown Date')
            
            if len(teams) >= 2:
                team1 = teams[0].get('name', 'Team 1')
                team2 = teams[1].get('name', 'Team 2')
                
                prompts.append({
                    "instruction": f"Tell me about the match between {team1} and {team2}",
                    "input": f"Venue: {venue}",
                    "output": f"The match between {team1} and {team2} was played at {venue} on {match_date}."
                })
        
        return prompts
    
    def load_all_data(self) -> List[Dict]:
        """Load all available data from all directories"""
        all_prompts = []
        
        print("=" * 60)
        print("Loading Batting Statistics...")
        print("=" * 60)
        batting_prompts = self.load_batting_stats()
        all_prompts.extend(batting_prompts)
        print(f"Total batting prompts: {len(batting_prompts)}\n")
        
        print("=" * 60)
        print("Loading Bowling Statistics...")
        print("=" * 60)
        bowling_prompts = self.load_bowling_stats()
        all_prompts.extend(bowling_prompts)
        print(f"Total bowling prompts: {len(bowling_prompts)}\n")
        
        print("=" * 60)
        print("Loading Match Information...")
        print("=" * 60)
        match_prompts = self.load_match_info()
        all_prompts.extend(match_prompts)
        print(f"Total match prompts: {len(match_prompts)}\n")
        
        print("=" * 60)
        print(f"TOTAL PROMPTS GENERATED: {len(all_prompts)}")
        print("=" * 60)
        
        return all_prompts


    


# In[3]:


class IPLLoRATrainer:
    """Fine-tune LLM on IPL dataset using LoRA"""
    
    def __init__(self, model_name: str = "gpt2", output_dir: str = "./ipl_lora_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA configuration"""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained( self.model_name, 
                                                            torch_dtype=torch.float16,
                                                            device_map="auto",
                                                            trust_remote_code=True
                                                         )
        
        # Configure LoRA
        lora_config = LoraConfig( r=8,  # Rank - can increase to 32 or 64 for better performance
                                  lora_alpha=32,  # Alpha scaling
                                  target_modules=["c_attn"] if "gpt2" in self.model_name else ["q_proj", "v_proj"],
                                  lora_dropout=0.05,
                                  bias="none",
                                   task_type=TaskType.CAUSAL_LM
                                )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        print("\n" + "="*60)
        self.model.print_trainable_parameters()
        print("="*60 + "\n")
        
    def prepare_dataset(self, prompts: List[Dict]) -> Dataset:
        """Convert prompts to tokenized dataset"""
        
        def format_prompt(example):
            """Format in instruction-following style"""
            text = f"""### Instruction:
                    {example['instruction']}
                    
                    ### Input:
                    {example['input']}
                    
                    ### Response:
                    {example['output']}"""
            return text
        
        # Format all prompts
        formatted_texts = [format_prompt(p) for p in prompts]
        
        # Tokenize
        encodings = self.tokenizer( formatted_texts,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=256,
                                    return_tensors=None
                                  )
        
        # Create dataset
        dataset_dict = { "input_ids": encodings["input_ids"],
                         "attention_mask": encodings["attention_mask"],
                         "labels" : encodings["input_ids"]
                       }
        
        return Dataset.from_dict(dataset_dict)
    
    def train(self, train_dataset: Dataset, epochs: int = 3, batch_size: int = 4):
        """Train the model with LoRA"""
        
        training_args = TrainingArguments( output_dir=self.output_dir,
                                            num_train_epochs=epochs,
                                            per_device_train_batch_size=1,
                                            gradient_accumulation_steps=4,
                                            learning_rate=2e-4,
                                            fp16=True,
                                            logging_steps=100,
                                            save_strategy="epoch",
                                            save_total_limit=2,
                                            warmup_ratio=0.03,
                                            lr_scheduler_type="cosine",
                                            report_to="none"
                                        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling( tokenizer=self.tokenizer,
                                                          mlm=False
                                                        )
        
        # Initialize trainer
        trainer = Trainer( model=self.model, 
                           args=training_args, 
                            train_dataset=train_dataset, 
                            data_collator=data_collator)
        
        # Train
        print("\n" + "="*60)
        print("Starting LoRA Training...")
        print("="*60 + "\n")
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"\n{'='*60}")
        print(f"Model saved to {self.output_dir}")
        print(f"{'='*60}\n")


# Main execution pipeline
if __name__ == "__main__":
    # Configuration
    IPL_DATA_PATH = "./IPL_DataSet_FineTuning/data/Indian_Premier_League_2022-03-26"  # Change this to your actual path
    MODEL_NAME = "gpt2"  # Use "meta-llama/Llama-2-7b-hf" for production
    OUTPUT_DIR = "./ipl_cricket_lora"
    EPOCHS = 3
    BATCH_SIZE = 4
    MAX_SAMPLES = 100 
    
    print("\n" + "="*60)
    print("IPL CRICKET DATASET - LoRA FINE-TUNING PIPELINE")
    print("="*60)
    print(f"Data Path: {IPL_DATA_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*60 + "\n")
    
    # 1. Process IPL Dataset
    processor = IPLJSONDatasetProcessor(base_path=IPL_DATA_PATH)
    all_prompts = processor.load_all_data()
    
    if not all_prompts:
        print("\n ERROR: No prompts generated. Check your data path and JSON files.")
        exit(1)
    
    # Limit samples if specified
    if MAX_SAMPLES and len(all_prompts) > MAX_SAMPLES:
        print(f"\n Limiting to {MAX_SAMPLES} samples for testing")
        all_prompts = all_prompts[:MAX_SAMPLES]

    # Show sample prompts
    print("\n" + "="*60)
    print("SAMPLE TRAINING PROMPTS")
    print("="*60)
    for i, prompt in enumerate(all_prompts[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Instruction: {prompt['instruction']}")
        print(f"Input: {prompt['input']}")
        print(f"Output: {prompt['output'][:100]}...")
    print("\n" + "="*60 + "\n")
    
    # 2. Setup and Train LoRA Model
    trainer = IPLLoRATrainer( model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    
    trainer.setup_model_and_tokenizer()
    
    # Prepare dataset
    print("Preparing training dataset...")
    train_dataset = trainer.prepare_dataset(all_prompts)
    print(f"Dataset prepared: {len(train_dataset)} samples\n")
    
    # Train
    trainer.train(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Total prompts trained: {len(all_prompts)}")
    print("="*60 + "\n")


# In[ ]:





# In[ ]:





# In[ ]:




