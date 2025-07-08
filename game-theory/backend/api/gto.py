from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from database.connection import get_db
from models.user import User
from models.gto_analysis import MarketData, GTOAnalysis, UserPortfolio, MarketAlert, AnalysisStatus
from api.auth import get_current_user
from services.gto_engine import gto_engine, GTOAnalysisInput
from services.market_data_service import market_data_service
from services.openai_service import openai_service
from services.gto_chat_service import gto_chat_service
from schemas.gto import (
    GTOAnalysisRequest, GTOAnalysisResponse, MarketDataRequest, MarketDataResponse,
    MarketListResponse, AnalysisListResponse, QuickAnalysisResponse, MonteCarloResponse,
    PortfolioRequest, PortfolioResponse, AlertRequest, AlertResponse,
    ChatMessageRequest, ChatMessageResponse, ChatHistoryRequest, ChatHistoryResponse,
    ChatSessionRequest, ChatSessionResponse
)
from typing import List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gto", tags=["GTO Strategy"])

@router.get("/markets", response_model=MarketListResponse)
async def get_markets(
    request: MarketDataRequest = Depends(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get available markets for analysis"""
    try:
        markets = []
        
        if request.market_type == "prediction_market":
            market_data = await market_data_service.get_prediction_markets(request.limit)
        elif request.market_type == "crypto":
            market_data = await market_data_service.get_crypto_markets(request.symbols)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported market type"
            )
        
        # Convert to response format
        for data in market_data:
            markets.append(MarketDataResponse(
                market_id=data.market_id,
                source=data.source,
                market_type=data.market_type,
                symbol=data.symbol,
                question=data.question,
                current_price=data.current_price,
                market_probability=data.market_probability,
                volume_24h=data.volume_24h,
                liquidity=data.liquidity,
                last_updated=data.last_updated
            ))
        
        return MarketListResponse(
            markets=markets,
            total_count=len(markets)
        )
        
    except Exception as e:
        logger.error(f"Error fetching markets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch market data"
        )

@router.post("/analyze/quick", response_model=QuickAnalysisResponse)
async def quick_analysis(
    request: GTOAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Perform quick GTO analysis without saving to database"""
    try:
        # Get fresh market data
        market_data = await market_data_service.get_market_by_id(
            request.market_id, 
            request.source
        )
        
        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Market not found"
            )
        
        if market_data.market_probability is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market probability not available"
            )
        
        # Prepare analysis input
        analysis_input = GTOAnalysisInput(
            user_probability=request.user_probability,
            market_probability=market_data.market_probability,
            bankroll=request.bankroll,
            risk_tolerance=request.risk_tolerance or 1.0,
            payout_ratio=request.payout_ratio or 1.0
        )
        
        # Perform GTO analysis
        result = gto_engine.analyze_binary_bet(analysis_input)
        
        # Generate quick explanation
        explanation = _generate_quick_explanation(result, market_data)
        
        return QuickAnalysisResponse(
            edge=result.edge,
            kelly_fraction=result.kelly_fraction,
            recommended_stake=result.recommended_stake,
            expected_value=result.expected_value,
            confidence_score=result.confidence_score,
            risk_assessment=result.risk_assessment,
            should_bet=result.should_bet,
            explanation_summary=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed"
        )

@router.post("/analyze", response_model=GTOAnalysisResponse)
async def create_analysis(
    request: GTOAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create full GTO analysis and save to database"""
    try:
        # Get fresh market data
        market_data = await market_data_service.get_market_by_id(
            request.market_id, 
            request.source
        )
        
        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Market not found"
            )
        
        # Save market data to database
        db_market_data = _save_market_data(db, market_data)
        
        # Create analysis record
        analysis = GTOAnalysis(
            user_id=current_user.id,
            market_data_id=db_market_data.id,
            user_probability=request.user_probability,
            bankroll=request.bankroll,
            risk_tolerance=request.risk_tolerance or 1.0,
            analysis_status=AnalysisStatus.PENDING
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        # Queue background analysis
        background_tasks.add_task(
            _complete_analysis,
            analysis.id,
            analysis_input=GTOAnalysisInput(
                user_probability=request.user_probability,
                market_probability=market_data.market_probability or 0.5,
                bankroll=request.bankroll,
                risk_tolerance=request.risk_tolerance or 1.0,
                payout_ratio=request.payout_ratio or 1.0
            )
        )
        
        return _analysis_to_response(analysis, db_market_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create analysis"
        )

@router.get("/analyses", response_model=AnalysisListResponse)
async def get_user_analyses(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's GTO analyses"""
    analyses = db.query(GTOAnalysis).filter(
        GTOAnalysis.user_id == current_user.id
    ).offset(offset).limit(limit).all()
    
    total_count = db.query(GTOAnalysis).filter(
        GTOAnalysis.user_id == current_user.id
    ).count()
    
    response_analyses = []
    for analysis in analyses:
        response_analyses.append(_analysis_to_response(analysis, analysis.market_data))
    
    return AnalysisListResponse(
        analyses=response_analyses,
        total_count=total_count
    )

@router.get("/analyses/{analysis_id}", response_model=GTOAnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific analysis by ID"""
    analysis = db.query(GTOAnalysis).filter(
        GTOAnalysis.id == analysis_id,
        GTOAnalysis.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return _analysis_to_response(analysis, analysis.market_data)

@router.post("/simulate", response_model=MonteCarloResponse)
async def monte_carlo_simulation(
    request: GTOAnalysisRequest,
    simulations: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """Run Monte Carlo simulation for strategy assessment"""
    try:
        # Get market data
        market_data = await market_data_service.get_market_by_id(
            request.market_id, 
            request.source
        )
        
        if not market_data or market_data.market_probability is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Market data not available"
            )
        
        # Prepare analysis input
        analysis_input = GTOAnalysisInput(
            user_probability=request.user_probability,
            market_probability=market_data.market_probability,
            bankroll=request.bankroll,
            risk_tolerance=request.risk_tolerance or 1.0,
            payout_ratio=request.payout_ratio or 1.0
        )
        
        # Run simulation
        simulation_result = gto_engine.monte_carlo_simulation(
            analysis_input, 
            num_simulations=min(simulations, 10000)  # Cap simulations
        )
        
        if "error" in simulation_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=simulation_result["error"]
            )
        
        return MonteCarloResponse(**simulation_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation failed"
        )

@router.post("/portfolio", response_model=PortfolioResponse)
async def create_portfolio(
    request: PortfolioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update user portfolio"""
    # Check if user already has a portfolio
    existing_portfolio = db.query(UserPortfolio).filter(
        UserPortfolio.user_id == current_user.id
    ).first()
    
    if existing_portfolio:
        # Update existing portfolio
        existing_portfolio.total_bankroll = request.total_bankroll
        existing_portfolio.available_bankroll = request.total_bankroll
        existing_portfolio.risk_preference = request.risk_preference
        existing_portfolio.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_portfolio)
        portfolio = existing_portfolio
    else:
        # Create new portfolio
        portfolio = UserPortfolio(
            user_id=current_user.id,
            total_bankroll=request.total_bankroll,
            available_bankroll=request.total_bankroll,
            risk_preference=request.risk_preference
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
    
    return PortfolioResponse(
        id=portfolio.id,
        total_bankroll=portfolio.total_bankroll,
        available_bankroll=portfolio.available_bankroll,
        risk_preference=portfolio.risk_preference,
        total_positions=portfolio.total_positions,
        total_at_risk=portfolio.total_at_risk,
        portfolio_expected_value=portfolio.portfolio_expected_value,
        portfolio_variance=portfolio.portfolio_variance,
        created_at=portfolio.created_at,
        updated_at=portfolio.updated_at
    )

@router.post("/chat", response_model=ChatMessageResponse)
async def gto_chat(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Interactive GTO strategy chat"""
    try:
        # Check rate limits for chat (implement basic rate limiting)
        # In production, you'd want proper rate limiting based on subscription tier
        
        response = await gto_chat_service.process_chat_message(
            user_id=current_user.id,
            message=request.message,
            analysis_id=request.analysis_id,
            session_id=request.session_id,
            db=db
        )
        
        if "error" in response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response["error"]
            )
        
        return ChatMessageResponse(
            response=response["response"],
            session_id=response["session_id"],
            suggestions=response.get("suggestions", []),
            analysis_referenced=response.get("analysis_referenced", False),
            timestamp=response["timestamp"],
            model_used=response.get("model_used")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GTO chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat service temporarily unavailable"
        )

@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for a session"""
    try:
        messages = gto_chat_service.get_chat_history(current_user.id, session_id)
        
        return ChatHistoryResponse(
            messages=[
                ChatHistoryMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    analysis_id=msg.get("analysis_id")
                )
                for msg in messages
            ],
            session_id=session_id,
            total_messages=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )

@router.post("/chat/session", response_model=ChatSessionResponse)
async def manage_chat_session(
    request: ChatSessionRequest,
    current_user: User = Depends(get_current_user)
):
    """Manage chat sessions (clear, list, etc.)"""
    try:
        if request.action == "clear":
            # Note: This would require session_id in a real implementation
            # For now, just return success
            return ChatSessionResponse(
                success=True,
                message="Chat session cleared successfully"
            )
        elif request.action == "list":
            # Return list of active sessions (placeholder)
            return ChatSessionResponse(
                success=True,
                message="Active sessions retrieved",
                active_sessions=[]  # Would be populated in real implementation
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid action. Use 'clear' or 'list'"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error managing chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session management failed"
        )

# Helper functions
def _save_market_data(db: Session, market_data) -> MarketData:
    """Save market data to database"""
    # Check if market data already exists
    existing = db.query(MarketData).filter(
        MarketData.market_id == market_data.market_id,
        MarketData.source == market_data.source
    ).first()
    
    if existing:
        # Update existing record
        existing.current_price = market_data.current_price
        existing.market_probability = market_data.market_probability
        existing.volume_24h = market_data.volume_24h
        existing.liquidity = market_data.liquidity
        existing.updated_at = datetime.utcnow()
        existing.raw_data = market_data.raw_data
        db.commit()
        return existing
    else:
        # Create new record
        db_market_data = MarketData(
            market_type=market_data.market_type,
            market_id=market_data.market_id,
            source=market_data.source,
            symbol=market_data.symbol,
            question=market_data.question,
            current_price=market_data.current_price,
            market_probability=market_data.market_probability,
            volume_24h=market_data.volume_24h,
            liquidity=market_data.liquidity,
            raw_data=market_data.raw_data
        )
        db.add(db_market_data)
        db.commit()
        db.refresh(db_market_data)
        return db_market_data

async def _complete_analysis(analysis_id: int, analysis_input: GTOAnalysisInput):
    """Complete GTO analysis in background"""
    try:
        from database.connection import SessionLocal
        db = SessionLocal()
        
        analysis = db.query(GTOAnalysis).filter(GTOAnalysis.id == analysis_id).first()
        if not analysis:
            return
        
        # Perform GTO calculation
        result = gto_engine.analyze_binary_bet(analysis_input)
        
        # Update analysis with results
        analysis.edge = result.edge
        analysis.kelly_fraction = result.kelly_fraction
        analysis.recommended_stake = result.recommended_stake
        analysis.expected_value = result.expected_value
        analysis.probability_of_loss = result.probability_of_loss
        analysis.max_loss = result.max_loss
        analysis.kelly_growth_rate = result.kelly_growth_rate
        analysis.analysis_status = AnalysisStatus.COMPLETED
        analysis.completed_at = datetime.utcnow()
        
        # Generate AI explanation
        explanation = await _generate_ai_explanation(result, analysis.market_data)
        analysis.ai_explanation = explanation
        
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Error completing analysis {analysis_id}: {str(e)}")

def _analysis_to_response(analysis: GTOAnalysis, market_data: MarketData) -> GTOAnalysisResponse:
    """Convert database analysis to response format"""
    market_response = MarketDataResponse(
        market_id=market_data.market_id,
        source=market_data.source,
        market_type=market_data.market_type.value,
        symbol=market_data.symbol,
        question=market_data.question,
        current_price=market_data.current_price,
        market_probability=market_data.market_probability,
        volume_24h=market_data.volume_24h,
        liquidity=market_data.liquidity,
        last_updated=market_data.updated_at
    )
    
    return GTOAnalysisResponse(
        id=analysis.id,
        market_data=market_response,
        user_probability=analysis.user_probability,
        bankroll=analysis.bankroll,
        edge=analysis.edge,
        kelly_fraction=analysis.kelly_fraction,
        recommended_stake=analysis.recommended_stake,
        expected_value=analysis.expected_value,
        probability_of_loss=analysis.probability_of_loss,
        max_loss=analysis.max_loss,
        kelly_growth_rate=analysis.kelly_growth_rate,
        analysis_status=analysis.analysis_status,
        ai_explanation=analysis.ai_explanation,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at
    )

def _generate_quick_explanation(result, market_data) -> str:
    """Generate quick explanation without AI"""
    if not result.should_bet:
        return f"No positive edge detected. Market probability ({market_data.market_probability:.1%}) is close to or higher than your belief ({result.edge + market_data.market_probability:.1%})."
    
    return f"Positive edge of {result.edge:.1%} detected. Recommended stake: ${result.recommended_stake:.2f} ({result.kelly_fraction:.1%} of bankroll). Expected value: ${result.expected_value:.2f}."

async def _generate_ai_explanation(result, market_data) -> str:
    """Generate detailed AI explanation"""
    try:
        prompt = f"""
        Explain this GTO betting strategy analysis in clear, educational terms:
        
        Market: {market_data.question or market_data.symbol}
        Market Probability: {market_data.market_probability:.1%}
        User's Probability: {(result.edge + market_data.market_probability):.1%}
        Edge: {result.edge:.1%}
        Recommended Stake: ${result.recommended_stake:.2f}
        Kelly Fraction: {result.kelly_fraction:.1%}
        Expected Value: ${result.expected_value:.2f}
        Risk Level: {result.risk_assessment}
        
        Explain the strategy, reasoning, and risk considerations in 2-3 paragraphs.
        Focus on education and include appropriate disclaimers.
        """
        
        # Use existing OpenAI service
        explanation = await openai_service.generate_trading_analysis(
            prompt, "base"  # Use base tier for explanations
        )
        
        if explanation["success"]:
            return explanation["analysis"]
        else:
            return _generate_quick_explanation(result, market_data)
            
    except Exception as e:
        logger.error(f"Error generating AI explanation: {str(e)}")
        return _generate_quick_explanation(result, market_data)