from openai import OpenAI
from typing import Dict, Any, List, Optional
import logging
from config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

class OpenAIService:
    def __init__(self):
        if not client:
            logger.warning("OpenAI API key not configured")
            
    def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connectivity with a simple request."""
        if not client:
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }
            
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a test assistant. Respond with a simple confirmation."
                    },
                    {
                        "role": "user", 
                        "content": "Please confirm you are working correctly."
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_trading_analysis(self, prompt: str, user_tier: str = "free") -> Dict[str, Any]:
        """Generate trading analysis based on user input."""
        if not client:
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }
            
        try:
            # Adjust model and parameters based on user tier
            model = "gpt-4" if user_tier in ["base", "premium"] else "gpt-3.5-turbo"
            max_tokens = 800 if user_tier == "premium" else 500 if user_tier == "base" else 300
            
            system_prompt = """You are an AI assistant specialized in educational trading analysis. Your role is to provide thoughtful, balanced analysis of trading scenarios for educational purposes only.

IMPORTANT GUIDELINES:
- This is for educational simulation only, not financial advice
- Always include disclaimers about risk and uncertainty
- Use probabilistic language (could, might, potentially) rather than certainties
- Provide reasoning based on historical patterns and general market knowledge
- Do not make specific price predictions
- Emphasize that real trading involves significant risk

Format your response with:
1. Scenario Analysis
2. Potential Outcomes (both positive and negative)
3. Key Factors to Consider
4. Educational Insights
5. Risk Disclaimer"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "user_tier": user_tier
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_gto_chat_response(
        self, 
        system_prompt: str, 
        messages: List[Dict[str, str]], 
        analysis_context: Optional[Any] = None,
        user_tier: str = "free"
    ) -> Dict[str, Any]:
        """Generate conversational GTO strategy response."""
        if not client:
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }
            
        try:
            # Adjust model based on user tier
            model = "gpt-4" if user_tier in ["base", "premium"] else "gpt-3.5-turbo"
            max_tokens = 600 if user_tier == "premium" else 400 if user_tier == "base" else 250
            
            # Build complete message chain
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages)
            
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.4  # Slightly higher for more conversational tone
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI GTO chat response failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_gto_explanation(
        self,
        analysis_data: Dict[str, Any],
        user_question: Optional[str] = None,
        user_tier: str = "base"
    ) -> Dict[str, Any]:
        """Generate detailed GTO analysis explanation."""
        if not client:
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }
            
        try:
            # Build explanation prompt
            prompt_parts = [
                "Explain this GTO betting strategy analysis in clear, educational terms:",
                "",
                f"Market: {analysis_data.get('market_question', analysis_data.get('symbol', 'Unknown'))}",
                f"Market Probability: {analysis_data.get('market_probability', 0):.1%}",
                f"User's Probability: {analysis_data.get('user_probability', 0):.1%}",
                f"Edge: {analysis_data.get('edge', 0):.1%}",
                f"Recommended Stake: ${analysis_data.get('recommended_stake', 0):.2f}",
                f"Kelly Fraction: {analysis_data.get('kelly_fraction', 0):.1%}",
                f"Expected Value: ${analysis_data.get('expected_value', 0):.2f}",
                f"Risk Level: {analysis_data.get('risk_assessment', 'Unknown')}",
                ""
            ]
            
            if user_question:
                prompt_parts.append(f"User's specific question: {user_question}")
                prompt_parts.append("")
            
            prompt_parts.extend([
                "Explain the strategy, reasoning, and risk considerations in 2-3 paragraphs.",
                "Focus on education and include appropriate disclaimers.",
                "Use clear language that helps the user understand the mathematical concepts."
            ])
            
            prompt = "\n".join(prompt_parts)
            
            # Select model based on tier
            model = "gpt-4" if user_tier in ["base", "premium"] else "gpt-3.5-turbo"
            max_tokens = 500 if user_tier == "premium" else 350
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Game Theory Optimal (GTO) strategy expert. Provide clear, educational explanations of betting strategies and mathematical concepts. Always emphasize that this is educational content and include appropriate risk disclaimers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "model": response.model,
                "user_tier": user_tier
            }
            
        except Exception as e:
            logger.error(f"OpenAI GTO explanation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Create singleton instance
openai_service = OpenAIService()