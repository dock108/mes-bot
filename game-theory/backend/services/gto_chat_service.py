import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from sqlalchemy.orm import Session
from models.gto_analysis import GTOAnalysis, MarketData
from models.user import User
from services.openai_service import openai_service

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    analysis_id: Optional[int] = None

@dataclass
class ChatContext:
    user_id: int
    conversation_history: List[ChatMessage]
    current_analysis: Optional[GTOAnalysis] = None
    session_id: Optional[str] = None

class GTOChatService:
    """Service for handling conversational GTO strategy guidance"""
    
    def __init__(self):
        # In-memory storage for chat sessions (in production, use Redis or database)
        self.chat_sessions: Dict[str, ChatContext] = {}
        self.max_history_length = 20  # Limit conversation history
    
    async def process_chat_message(
        self, 
        user_id: int,
        message: str,
        analysis_id: Optional[int] = None,
        session_id: Optional[str] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """Process a chat message and return AI response with context"""
        
        try:
            # Get or create chat context
            context = self._get_or_create_context(user_id, session_id, analysis_id, db)
            
            # Add user message to history
            user_message = ChatMessage(
                role="user",
                content=message,
                timestamp=datetime.utcnow(),
                analysis_id=analysis_id
            )
            context.conversation_history.append(user_message)
            
            # Generate AI response
            ai_response = await self._generate_ai_response(context, db)
            
            # Add AI response to history
            assistant_message = ChatMessage(
                role="assistant",
                content=ai_response["content"],
                timestamp=datetime.utcnow(),
                analysis_id=analysis_id
            )
            context.conversation_history.append(assistant_message)
            
            # Trim history if too long
            self._trim_conversation_history(context)
            
            # Save context back to session
            if context.session_id:
                self.chat_sessions[context.session_id] = context
            
            return {
                "response": ai_response["content"],
                "session_id": context.session_id,
                "suggestions": ai_response.get("suggestions", []),
                "analysis_referenced": analysis_id is not None,
                "timestamp": assistant_message.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error processing your question. Please try again.",
                "error": str(e),
                "session_id": session_id
            }
    
    def _get_or_create_context(
        self, 
        user_id: int, 
        session_id: Optional[str], 
        analysis_id: Optional[int],
        db: Session
    ) -> ChatContext:
        """Get existing chat context or create new one"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"chat_{user_id}_{datetime.utcnow().timestamp()}"
        
        # Try to get existing context
        if session_id in self.chat_sessions:
            context = self.chat_sessions[session_id]
        else:
            # Create new context
            context = ChatContext(
                user_id=user_id,
                conversation_history=[],
                session_id=session_id
            )
        
        # Load analysis if provided
        if analysis_id and db:
            analysis = db.query(GTOAnalysis).filter(
                GTOAnalysis.id == analysis_id,
                GTOAnalysis.user_id == user_id
            ).first()
            context.current_analysis = analysis
        
        return context
    
    async def _generate_ai_response(self, context: ChatContext, db: Session) -> Dict[str, Any]:
        """Generate AI response using OpenAI with GTO context"""
        
        # Build conversation prompt with context
        system_prompt = self._build_system_prompt(context)
        conversation_messages = self._build_conversation_messages(context)
        
        # Generate response using OpenAI service
        result = await openai_service.generate_gto_chat_response(
            system_prompt=system_prompt,
            messages=conversation_messages,
            analysis_context=context.current_analysis
        )
        
        if result["success"]:
            # Generate follow-up suggestions
            suggestions = self._generate_suggestions(context)
            
            return {
                "content": result["response"],
                "suggestions": suggestions,
                "model": result.get("model", "gpt-4")
            }
        else:
            raise Exception(f"AI response generation failed: {result.get('error', 'Unknown error')}")
    
    def _build_system_prompt(self, context: ChatContext) -> str:
        """Build system prompt for GTO chat"""
        
        base_prompt = """You are a Game Theory Optimal (GTO) strategy expert and educational tutor. 
Your role is to help users understand their betting analysis results and improve their strategic thinking.

Key principles:
1. Always provide educational value - explain the 'why' behind recommendations
2. Be encouraging but realistic about risks
3. Reference Kelly Criterion, edge calculations, and risk management principles
4. Suggest follow-up questions to deepen understanding
5. Always emphasize that this is educational content, not financial advice

Communication style:
- Clear, conversational, and encouraging
- Use analogies and examples when explaining complex concepts
- Break down mathematical concepts into understandable terms
- Ask clarifying questions when the user's intent is unclear"""

        # Add analysis context if available
        if context.current_analysis:
            analysis = context.current_analysis
            analysis_context = f"""

Current Analysis Context:
- Market: {analysis.market_data.question or analysis.market_data.symbol}
- Your Probability Estimate: {analysis.user_probability:.1%}
- Market Probability: {analysis.market_data.market_probability:.1%}
- Calculated Edge: {analysis.edge:.1%}
- Kelly Fraction: {analysis.kelly_fraction:.1%}
- Recommended Stake: ${analysis.recommended_stake:.2f}
- Expected Value: ${analysis.expected_value:.2f}
- Risk Assessment: {getattr(analysis, 'risk_assessment', 'Not calculated')}

Use this context to provide specific, relevant guidance about this analysis."""
            
            base_prompt += analysis_context
        
        return base_prompt
    
    def _build_conversation_messages(self, context: ChatContext) -> List[Dict[str, str]]:
        """Build conversation messages for OpenAI API"""
        
        messages = []
        
        # Add recent conversation history (last 10 messages to stay within token limits)
        recent_history = context.conversation_history[-10:] if len(context.conversation_history) > 10 else context.conversation_history
        
        for msg in recent_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def _generate_suggestions(self, context: ChatContext) -> List[str]:
        """Generate follow-up question suggestions"""
        
        base_suggestions = [
            "Can you explain the Kelly Criterion in simple terms?",
            "What factors should I consider when estimating probabilities?",
            "How do I manage risk in my betting strategy?",
            "What does 'edge' mean in betting?"
        ]
        
        # Context-specific suggestions
        if context.current_analysis:
            analysis = context.current_analysis
            
            if analysis.edge > 0.05:  # Strong edge
                base_suggestions.insert(0, "Why is this considered a strong betting opportunity?")
                base_suggestions.insert(1, "What could go wrong with this bet?")
            elif analysis.edge <= 0:  # No edge
                base_suggestions.insert(0, "Why shouldn't I bet on this market?")
                base_suggestions.insert(1, "How can I find better opportunities?")
            
            if analysis.kelly_fraction > 0.2:  # High Kelly
                base_suggestions.insert(0, "Is betting 20%+ of my bankroll too risky?")
            
            if hasattr(analysis, 'risk_assessment') and analysis.risk_assessment == 'high':
                base_suggestions.insert(0, "How can I reduce the risk of this bet?")
        
        return base_suggestions[:4]  # Return top 4 suggestions
    
    def _trim_conversation_history(self, context: ChatContext):
        """Trim conversation history to prevent memory issues"""
        if len(context.conversation_history) > self.max_history_length:
            # Keep the most recent messages
            context.conversation_history = context.conversation_history[-self.max_history_length:]
    
    def get_chat_history(self, user_id: int, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        
        if session_id not in self.chat_sessions:
            return []
        
        context = self.chat_sessions[session_id]
        if context.user_id != user_id:
            return []  # Security check
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "analysis_id": msg.analysis_id
            }
            for msg in context.conversation_history
        ]
    
    def clear_chat_session(self, user_id: int, session_id: str) -> bool:
        """Clear a chat session"""
        
        if session_id in self.chat_sessions:
            context = self.chat_sessions[session_id]
            if context.user_id == user_id:
                del self.chat_sessions[session_id]
                return True
        
        return False

# Global service instance
gto_chat_service = GTOChatService()