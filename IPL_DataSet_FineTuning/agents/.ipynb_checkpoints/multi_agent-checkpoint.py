"""
FastAPI Inference Service with Multi-Agent Flow
================================================

Multi-Agent Architecture:
- Agent 1 (Retriever): Fetches exact cricket statistics from database
- Agent 2 (Analyzer): LLM analyzes and compares stats
- Agent 3 (Explainer): LLM provides natural language explanation
- Coordinator: Routes queries to appropriate agents and combines results

Installation:
pip install fastapi uvicorn transformers peft torch accelerate huggingface_hub python-multipart
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from huggingface_hub import login
import logging
from datetime import datetime
import os
from functools import lru_cache
import time
import re
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    
    # Model Configuration
    BASE_MODEL = "gpt2"
    LORA_MODEL_REPO = "your-username/ipl-cricket-lora"
    HF_TOKEN = os.getenv("HF_TOKEN", "hf_your_token_here")
    
    # API Configuration
    API_TITLE = "IPL Cricket Multi-Agent API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = """
      IPL Cricket Statistics Multi-Agent API
    
    This API uses a multi-agent system for intelligent cricket queries:
    
    ## Agents
    1. **Retriever Agent**: Fetches exact statistics from database
    2. **Analyzer Agent**: Performs calculations and comparisons
    3. **Explainer Agent**: Provides natural language explanations
    4. **Coordinator**: Routes queries to appropriate agents
    
    ## Features
    - Exact statistics retrieval
    - Complex statistical analysis
    - Player comparisons
    - Natural language explanations
    - Multi-agent collaboration
    """
    
    # Generation Configuration
    MAX_NEW_TOKENS = 200
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    
    # Security
    API_KEYS = ["demo-key-12345", "prod-key-67890"]
    
    # Performance
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE_SIZE = 100


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AgentType(str, Enum):
    """Available agent types"""
    RETRIEVER = "retriever"
    ANALYZER = "analyzer"
    EXPLAINER = "explainer"
    Coordinator = "coordinator"


class QueryRequest(BaseModel):
    """Request model for cricket statistics query"""
    
    query: str = Field( ...,
                        description="Natural language query about IPL cricket",
                        example="Compare Virat Kohli and Rohit Sharma batting averages in IPL 2022"
                      )
    
    agent_mode: Optional[Literal["auto", "retriever", "analyzer", "explainer"]] = Field( "auto",
                                    description="Which agent to use (auto = coordinator decides)"
                                    )
    
    include_agent_trace: bool = Field( False,
                                        description="Include detailed agent execution trace"
                                     )   
    max_tokens: Optional[int] = Field(None, ge=50, le=500)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class AgentExecution(BaseModel):
    """Agent execution details"""
    agent_name: str
    input: str
    output: str
    execution_time_ms: float
    confidence: float


class QueryResponse(BaseModel):
    """Response model with agent information"""
    query: str
    answer: str
    confidence: float
    agents_used: List[str]
    timestamp: str
    processing_time_ms: float
    agent_trace: Optional[List[AgentExecution]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    agents_available: List[str]
    device: str
    timestamp: str
    version: str


# ============================================================================
# MOCK STATISTICS DATABASE (Replace with real database)
# ============================================================================

class StatsDatabase:
    """
    Mock statistics database
    In production, replace with real database queries
    """
    
    # Sample IPL 2022 statistics
    STATS = {
        "virat_kohli": {
            "player_name": "Virat Kohli",
            "team": "Royal Challengers Bangalore",
            "matches": 16,
            "innings": 16,
            "runs": 341,
            "average": 22.73,
            "strike_rate": 115.99,
            "highest_score": 73,
            "centuries": 0,
            "fifties": 2,
            "fours": 32,
            "sixes": 8
        },
        "rohit_sharma": {
            "player_name": "Rohit Sharma",
            "team": "Mumbai Indians",
            "matches": 14,
            "innings": 14,
            "runs": 268,
            "average": 19.14,
            "strike_rate": 120.18,
            "highest_score": 48,
            "centuries": 0,
            "fifties": 0,
            "fours": 28,
            "sixes": 13
        },
        "ms_dhoni": {
            "player_name": "MS Dhoni",
            "team": "Chennai Super Kings",
            "matches": 14,
            "innings": 13,
            "runs": 232,
            "average": 33.14,
            "strike_rate": 123.40,
            "highest_score": 50,
            "centuries": 0,
            "fifties": 1,
            "fours": 21,
            "sixes": 10
        },
        "jos_buttler": {
            "player_name": "Jos Buttler",
            "team": "Rajasthan Royals",
            "matches": 17,
            "innings": 17,
            "runs": 863,
            "average": 57.53,
            "strike_rate": 149.05,
            "highest_score": 116,
            "centuries": 4,
            "fifties": 4,
            "fours": 83,
            "sixes": 45
        },
        "david_miller": {
            "player_name": "David Miller",
            "team": "Gujarat Titans",
            "matches": 16,
            "innings": 16,
            "runs": 481,
            "average": 68.71,
            "strike_rate": 142.73,
            "highest_score": 94,
            "centuries": 0,
            "fifties": 2,
            "fours": 32,
            "sixes": 23
        }
    }
    
    @classmethod
    def get_player_stats(cls, player_name: str) -> Optional[Dict]:
        """Get statistics for a player"""
        # Normalize name
        normalized = player_name.lower().replace(" ", "_")
        return cls.STATS.get(normalized)
    
    @classmethod
    def search_players(cls, query: str) -> List[str]:
        """Search for players matching query"""
        query_lower = query.lower()
        matches = []
        
        for key, stats in cls.STATS.items():
            if query_lower in key or query_lower in stats["player_name"].lower():
                matches.append(stats["player_name"])
        
        return matches


# ============================================================================
# AGENT: RETRIEVER
# ============================================================================

class RetrieverAgent:
    """
    Agent 1: Retrieves exact statistics from database
    Handles: "What are X's stats?", "Get stats for Y"
    """
    
    def __init__(self):
        self.name = "Retriever Agent"
    
    def can_handle(self, query: str) -> bool:
        """Determine if this agent can handle the query"""
        patterns = [
            r"what (are|is|were)",
            r"get stats",
            r"show.*statistics",
            r"batting.*statistics",
            r"bowling.*statistics",
            r"^stats for"
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def extract_player_name(self, query: str) -> Optional[str]:
        """Extract player name from query"""
        # Simple pattern matching (can be improved with NER)
        players = StatsDatabase.search_players(query)      
        if players:
            return players[0]  # Return first match
        return None
    
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute retrieval
        Returns: {success, data, message}
        """
        start_time = time.time()
        
        logger.info(f"{self.name}: Processing query")
        
        # Extract player name
        player_name = self.extract_player_name(query)
        
        if not player_name:
            return {
                "success": False,
                "data": None,
                "message": f"Could not identify player in query: {query}",
                "execution_time": time.time() - start_time
            }
        
        # Get stats
        stats = StatsDatabase.get_player_stats(player_name)
        
        if not stats:
            return {
                "success": False,
                "data": None,
                "message": f"No statistics found for {player_name}",
                "execution_time": time.time() - start_time
            }
        
        # Format response
        response = self._format_stats(stats)
        
        execution_time = time.time() - start_time
        
        logger.info(f" {self.name}: Retrieved stats in {execution_time:.3f}s")
        
        return {
            "success": True,
            "data": stats,
            "message": response,
            "execution_time": execution_time,
            "confidence": 1.0  # Perfect confidence for database retrieval
        }
    
    def _format_stats(self, stats: Dict) -> str:
        """Format statistics into readable text"""
        return f"""{stats['player_name']} - IPL 2022 Statistics:
                    Team: {stats['team']}
                    Matches: {stats['matches']}
                    Runs: {stats['runs']}
                    Average: {stats['average']}
                    Strike Rate: {stats['strike_rate']}
                    Highest Score: {stats['highest_score']}
                    Centuries: {stats['centuries']}
                    Fifties: {stats['fifties']}
                    Fours: {stats['fours']}
                    Sixes: {stats['sixes']}"""


# ============================================================================
# AGENT: ANALYZER
# ============================================================================

class AnalyzerAgent:
    """
    Agent 2: Performs statistical analysis and comparisons
    Handles: "Compare X and Y", "Who is better?", calculations
    """
    
    def __init__(self):
        self.name = "Analyzer Agent"
        self.retriever = RetrieverAgent()
    
    def can_handle(self, query: str) -> bool:
        """Determine if this agent can handle the query"""
        patterns = [
            r"compare",
            r"who is better",
            r"who has (higher|lower|more|better)",
            r"difference between",
            r"versus",
            r" vs ",
            r"which player"
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def extract_players(self, query: str) -> List[str]:
        """Extract multiple player names from comparison query"""
        # Get all matching players
        all_players = []
        
        for player_key, stats in StatsDatabase.STATS.items():
            player_name = stats["player_name"]
            if player_name.lower() in query.lower():
                all_players.append(player_name)
        
        return all_players
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute analysis"""
        start_time = time.time()
        
        logger.info(f"{self.name}: Processing query")
        
        # Extract players to compare
        players = self.extract_players(query)
        
        if len(players) < 2:
            return {
                "success": False,
                "data": None,
                "message": "Need at least 2 players to compare",
                "execution_time": time.time() - start_time
            }
        
        # Get stats for all players
        players_stats = []
        for player in players:
            stats = StatsDatabase.get_player_stats(player)
            if stats:
                players_stats.append(stats)
        
        if len(players_stats) < 2:
            return {
                "success": False,
                "data": None,
                "message": "Could not find stats for enough players",
                "execution_time": time.time() - start_time
            }
        
        # Perform comparison
        analysis = self._compare_players(players_stats, query)
        
        execution_time = time.time() - start_time
        
        logger.info(f"{self.name}: Analysis complete in {execution_time:.3f}s")
        
        return {
            "success": True,
            "data": players_stats,
            "message": analysis,
            "execution_time": execution_time,
            "confidence": 0.95
        }
    
    def _compare_players(self, players_stats: List[Dict], query: str) -> str:
        """Compare player statistics"""
        
        # Determine what metric to compare
        metric = self._identify_metric(query)
        
        comparison = f"Comparison of {len(players_stats)} players:\n\n"
        
        for stats in players_stats:
            comparison += f"{stats['player_name']}:\n"
            comparison += f"  Runs: {stats['runs']}, Average: {stats['average']}, Strike Rate: {stats['strike_rate']}\n"
        
        # Find best performer
        if metric == "runs":
            best = max(players_stats, key=lambda x: x['runs'])
            comparison += f"\n Highest run scorer: {best['player_name']} with {best['runs']} runs"
        
        elif metric == "average":
            best = max(players_stats, key=lambda x: x['average'])
            comparison += f"\n Best average: {best['player_name']} with {best['average']}"
        
        elif metric == "strike_rate":
            best = max(players_stats, key=lambda x: x['strike_rate'])
            comparison += f"\n Best strike rate: {best['player_name']} with {best['strike_rate']}"
        
        else:
            # General comparison
            best_runs = max(players_stats, key=lambda x: x['runs'])
            best_avg = max(players_stats, key=lambda x: x['average'])
            comparison += f"\n Most runs: {best_runs['player_name']}"
            comparison += f"\n Best average: {best_avg['player_name']}"
        
        return comparison
    
    def _identify_metric(self, query: str) -> str:
        """Identify which metric to focus on"""
        query_lower = query.lower()
        
        if "average" in query_lower:
            return "average"
        elif "strike rate" in query_lower or "strike" in query_lower:
            return "strike_rate"
        elif "runs" in query_lower:
            return "runs"
        else:
            return "general"


# ============================================================================
# AGENT: EXPLAINER (Uses LLM)
# ============================================================================

class ExplainerAgent:
    """
    Agent 3: Provides natural language explanations using LLM
    Handles: "Explain", "Why", "How does", analytical questions
    """
    
    def __init__(self, model_manager):
        self.name = "Explainer Agent"
        self.model_manager = model_manager
    
    def can_handle(self, query: str) -> bool:
        """Determine if this agent can handle the query"""
        patterns = [
            r"explain",
            r"why",
            r"how does",
            r"analyze",
            r"what does.*mean",
            r"tell me about"
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def execute(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute explanation using LLM
        Context can include data from other agents
        """
        start_time = time.time()        
        logger.info(f" {self.name}: Processing query")
        
        # Build enhanced prompt with context
        prompt = self._build_prompt(query, context)        
        # Generate explanation using LLM
        try:
            answer, confidence, metadata = self.model_manager.generate( query=prompt,
                                                                        max_tokens=200,
                                                                        temperature=0.8
                                                                      )           
            execution_time = time.time() - start_time           
            logger.info(f" {self.name}: Explanation generated in {execution_time:.3f}s")
            
            return { "success": True,
                    "data": {"answer": answer, "metadata": metadata},
                    "message": answer,
                    "execution_time": execution_time,
                    "confidence": confidence
                  }
            
        except Exception as e:
            logger.error(f" {self.name} failed: {str(e)}")
            return {
                "success": False,
                "data": None,
                "message": f"Failed to generate explanation: {str(e)}",
                "execution_time": time.time() - start_time
            }
    
    def _build_prompt(self, query: str, context: Optional[Dict]) -> str:
        """Build enhanced prompt with context"""
        
        if context:
            prompt = f"Based on the following statistics:\n\n"
            
            if "data" in context:
                # Include data from previous agents
                data = context["data"]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "player_name" in item:
                            prompt += f"{item['player_name']}: {item['runs']} runs, "
                            prompt += f"average {item['average']}, "
                            prompt += f"strike rate {item['strike_rate']}\n"
                
                elif isinstance(data, dict) and "player_name" in data:
                    prompt += f"{data['player_name']}: {data['runs']} runs, "
                    prompt += f"average {data['average']}, "
                    prompt += f"strike rate {data['strike_rate']}\n"
            
            prompt += f"\n{query}"
        else:
            prompt = query
        
        return prompt


# ============================================================================
# Multi-Agent Coordinator
# ============================================================================

class CoordinatorAgent:
    """
    Coordinator: Coordinates multiple agents
    Decides which agent(s) to use and combines their results
    """
    
    def __init__(self, model_manager):
        self.name = "Coordinator"
        self.retriever = RetrieverAgent()
        self.analyzer = AnalyzerAgent()
        self.explainer = ExplainerAgent(model_manager)
        
        self.execution_trace = []
    
    def execute(self, query: str, agent_mode: str = "auto") -> Dict[str, Any]:
        """
        Execute query using appropriate agent(s)
        """
        start_time = time.time()
        self.execution_trace = []
        
        logger.info(f"{self.name}: Routing query...")
        
        # If specific agent requested
        if agent_mode != "auto":
            return self._execute_single_agent(query, agent_mode)
        
        # Auto mode: determine workflow
        workflow = self._plan_workflow(query)
        
        logger.info(f"Workflow: {' → '.join(workflow)}")
        
        # Execute workflow
        result = self._execute_workflow(query, workflow)        
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        result["agents_used"] = workflow
        result["agent_trace"] = self.execution_trace
        return result

    
    def _plan_workflow(self, query: str) -> List[str]:
        """
        Plan which agents to use based on query
        """
        workflow = []       
        # Check each agent
        if self.analyzer.can_handle(query):
            # Comparison queries: Analyzer → Explainer
            workflow = ["analyzer", "explainer"]
        elif self.retriever.can_handle(query):
            # Simple stat queries: Retriever → Explainer
            workflow = ["retriever", "explainer"]
        elif self.explainer.can_handle(query):
            # Explanation queries: Explainer only
            workflow = ["explainer"]
        else:
            # Default: Try retriever first, then explainer
            workflow = ["retriever", "explainer"]       
        return workflow

    
    def _execute_workflow(self, query: str, workflow: List[str]) -> Dict[str, Any]:
        """Execute the planned workflow"""
        
        context = None
        final_answer = ""
        agents_used = []
        total_confidence = 0
        
        for agent_name in workflow:
            agent_start = time.time()
            
            # Execute agent
            if agent_name == "retriever":
                result = self.retriever.execute(query)
            
            elif agent_name == "analyzer":
                result = self.analyzer.execute(query)
            
            elif agent_name == "explainer":
                result = self.explainer.execute(query, context)
            
            else:
                continue
            
            agent_time = (time.time() - agent_start) * 1000
            
            # Record execution
            self.execution_trace.append(AgentExecution( agent_name=agent_name,
                                                        input=query if not context else f"{query} [with context]",
                                                        output=result.get("message", ""),
                                                        execution_time_ms=round(agent_time, 2),
                                                        confidence=result.get("confidence", 0.0)
                                                    ))
                                                        
            if result["success"]:
                # Update context for next agent
                context = result
                final_answer = result["message"]
                agents_used.append(agent_name)
                total_confidence += result.get("confidence", 0)
            else:
                logger.warning(f" {agent_name} failed: {result['message']}")
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(agents_used) if agents_used else 0
        
        return {
            "success": len(agents_used) > 0,
            "answer": final_answer,
            "confidence": avg_confidence,
            "agents_used": agents_used
        }
    
    def _execute_single_agent(self, query: str, agent_mode: str) -> Dict[str, Any]:
        """Execute a specific agent"""
        
        if agent_mode == "retriever":
            result = self.retriever.execute(query)
        elif agent_mode == "analyzer":
            result = self.analyzer.execute(query)
        elif agent_mode == "explainer":
            result = self.explainer.execute(query)
        else:
            return {
                "success": False,
                "answer": f"Unknown agent: {agent_mode}",
                "confidence": 0.0,
                "agents_used": []
            }
        
        return {
            "success": result["success"],
            "answer": result["message"],
            "confidence": result.get("confidence", 0.0),
            "agents_used": [agent_mode]
        }


# ============================================================================
# MODEL MANAGER (Same as before, with minor updates)
# ============================================================================

class ModelManager:
    """Manages model loading and inference"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _generation_config = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Load model from HuggingFace"""
        
        if self._initialized:
            return
        
        try:
            logger.info("=" * 60)
            logger.info("INITIALIZING MODEL")
            logger.info("=" * 60)
            
            login(token=Config.HF_TOKEN)
            
            self._tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "right"
            
            base_model = AutoModelForCausalLM.from_pretrained( Config.BASE_MODEL,
                                                                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                                                                device_map="auto" if Config.DEVICE == "cuda" else None,
                                                                trust_remote_code=True
                                                            )
            
            self._model = PeftModel.from_pretrained(  base_model,
                                                      Config.LORA_MODEL_REPO,
                                                      token=Config.HF_TOKEN
                                                   )
            
            self._model.eval()
            
            if Config.DEVICE == "cpu":
                self._model = self._model.to(Config.DEVICE)
            
            self._generation_config = GenerationConfig( max_new_tokens=Config.MAX_NEW_TOKENS,
                                                        temperature=Config.TEMPERATURE,
                                                        top_p=Config.TOP_P,
                                                        top_k=Config.TOP_K,
                                                        do_sample=True,
                                                        pad_token_id=self._tokenizer.pad_token_id,
                                                        eos_token_id=self._tokenizer.eos_token_id,
                                                    )
            
            self._initialized = True
            
            logger.info("MODEL LOADED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    @lru_cache(maxsize=Config.CACHE_SIZE)
    def generate( self,
                  query: str,
                  max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None
                ) -> tuple[str, float, Dict[str, Any]]:
            """Generate answer for query"""
        
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            prompt = f"""### Instruction:
                        {query}
                        
                        ### Input:
                        Format: T20
                        ### Response:
                        """            
            inputs = self._tokenizer( prompt,
                                      return_tensors="pt",
                                      truncation=True,
                                      max_length=512
                                    )
            
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            gen_config = self._generation_config
            if max_tokens or temperature:
                gen_config = GenerationConfig( max_new_tokens=max_tokens or Config.MAX_NEW_TOKENS,
                                                temperature=temperature or Config.TEMPERATURE,
                                                top_p=Config.TOP_P,
                                                top_k=Config.TOP_K,
                                                do_sample=True,
                                                pad_token_id=self._tokenizer.pad_token_id,
                                                eos_token_id=self._tokenizer.eos_token_id,
                                            )
            
            with torch.no_grad():
                outputs = self._model.generate( **inputs,
                                                 generation_config=gen_config,
                                                 return_dict_in_generate=True,
                                                 output_scores=True
                                                )
            
            generated_text = self._tokenizer.decode( outputs.sequences[0],
                                                     skip_special_tokens=True
                                                    )
            
            if "### Response:" in generated_text:
                answer = generated_text.split("### Response:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            confidence = 0.8
            
            generation_time = time.time() - start_time
            metadata = { "model": Config.BASE_MODEL,
                          "tokens_generated": len(outputs.sequences[0]) - inputs['input_ids'].shape[1],
                          "generation_time_seconds": round(generation_time, 3),
                          "device": Config.DEVICE,
                        }
            
            return answer, confidence, metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None


# ============================================================================
# SECURITY
# ============================================================================

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in Config.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware( CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                   )

model_manager = ModelManager()
coordinator = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global coordinator
    
    logger.info(" Starting Multi-Agent Cricket API...")
    
    try:
        model_manager.initialize()
        coordinator = CoordinatorAgent(model_manager)
        logger.info(" All agents initialized")
    except Exception as e:
        logger.error(f" Startup failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(" Shutting down Multi-Agent API...")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "IPL Cricket Multi-Agent API",
        "version": Config.API_VERSION,
        "agents": ["retriever", "analyzer", "explainer", "coordinator"],
        "docs": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse
)
async def health_check():
    return HealthResponse(
        status="healthy"
        