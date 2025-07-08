import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GTOAnalysisInput:
    """Input parameters for GTO analysis"""
    user_probability: float  # User's belief about outcome probability (0-1)
    market_probability: float  # Market-implied probability (0-1)
    bankroll: float  # Total available capital
    risk_tolerance: float = 1.0  # Kelly fraction multiplier (0-1)
    payout_ratio: float = 1.0  # Payout ratio for winning bet
    market_type: str = "binary"  # "binary", "continuous", "multi_outcome"

@dataclass
class GTOAnalysisResult:
    """Results from GTO analysis"""
    edge: float  # User's advantage over market
    kelly_fraction: float  # Optimal bet size as fraction of bankroll
    recommended_stake: float  # Dollar amount to bet
    expected_value: float  # Expected profit/loss
    probability_of_loss: float  # Chance of losing the bet
    max_loss: float  # Maximum possible loss
    kelly_growth_rate: float  # Expected growth rate using Kelly
    confidence_score: float  # Confidence in the recommendation (0-1)
    risk_assessment: str  # "low", "medium", "high"
    should_bet: bool  # Whether to make the bet

class GTOEngine:
    """Game Theory Optimal strategy engine for betting and trading decisions"""
    
    def __init__(self):
        self.min_edge_threshold = 0.01  # Minimum edge required to recommend betting
        self.max_kelly_fraction = 0.5  # Maximum Kelly fraction for safety
        self.min_bankroll = 1.0  # Minimum bankroll required
        
    def analyze_binary_bet(self, analysis_input: GTOAnalysisInput) -> GTOAnalysisResult:
        """
        Analyze a binary betting opportunity using Kelly Criterion
        
        Args:
            analysis_input: Input parameters for analysis
            
        Returns:
            GTOAnalysisResult with recommended strategy
        """
        try:
            # Input validation
            self._validate_input(analysis_input)
            
            # Calculate edge (user advantage)
            edge = self._calculate_edge(
                analysis_input.user_probability,
                analysis_input.market_probability,
                analysis_input.payout_ratio
            )
            
            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(
                analysis_input.user_probability,
                analysis_input.market_probability,
                analysis_input.payout_ratio
            )
            
            # Apply risk tolerance and safety limits
            adjusted_kelly = min(
                kelly_fraction * analysis_input.risk_tolerance,
                self.max_kelly_fraction
            )
            
            # Calculate recommended stake
            recommended_stake = max(0, adjusted_kelly * analysis_input.bankroll)
            
            # Calculate expected value
            expected_value = self._calculate_expected_value(
                analysis_input.user_probability,
                recommended_stake,
                analysis_input.payout_ratio
            )
            
            # Calculate risk metrics
            probability_of_loss = 1 - analysis_input.user_probability
            max_loss = recommended_stake
            kelly_growth_rate = self._calculate_kelly_growth_rate(
                analysis_input.user_probability,
                kelly_fraction,
                analysis_input.payout_ratio
            )
            
            # Determine confidence and risk assessment
            confidence_score = self._calculate_confidence_score(edge, kelly_fraction)
            risk_assessment = self._assess_risk_level(adjusted_kelly, edge)
            
            # Decision logic
            should_bet = (
                edge > self.min_edge_threshold and
                recommended_stake >= 1.0 and
                adjusted_kelly > 0
            )
            
            return GTOAnalysisResult(
                edge=edge,
                kelly_fraction=adjusted_kelly,
                recommended_stake=recommended_stake,
                expected_value=expected_value,
                probability_of_loss=probability_of_loss,
                max_loss=max_loss,
                kelly_growth_rate=kelly_growth_rate,
                confidence_score=confidence_score,
                risk_assessment=risk_assessment,
                should_bet=should_bet
            )
            
        except Exception as e:
            logger.error(f"Error in GTO analysis: {str(e)}")
            raise
    
    def _validate_input(self, analysis_input: GTOAnalysisInput) -> None:
        """Validate input parameters"""
        if not (0 <= analysis_input.user_probability <= 1):
            raise ValueError("User probability must be between 0 and 1")
        
        if not (0 <= analysis_input.market_probability <= 1):
            raise ValueError("Market probability must be between 0 and 1")
        
        if analysis_input.bankroll < self.min_bankroll:
            raise ValueError(f"Bankroll must be at least ${self.min_bankroll}")
        
        if not (0 < analysis_input.risk_tolerance <= 1):
            raise ValueError("Risk tolerance must be between 0 and 1")
        
        if analysis_input.payout_ratio <= 0:
            raise ValueError("Payout ratio must be positive")
    
    def _calculate_edge(self, user_prob: float, market_prob: float, payout_ratio: float) -> float:
        """Calculate user's edge over the market"""
        # For binary bet: edge = user_prob * (payout_ratio + 1) - 1
        # Simplified for equal odds: edge = user_prob - market_prob
        if payout_ratio == 1.0:
            return user_prob - market_prob
        else:
            # More complex calculation for non-equal odds
            fair_odds = (1 - market_prob) / market_prob
            implied_payout = fair_odds
            expected_return = user_prob * payout_ratio - (1 - user_prob)
            market_expected = market_prob * implied_payout - (1 - market_prob)
            return expected_return - market_expected
    
    def _calculate_kelly_fraction(self, user_prob: float, market_prob: float, payout_ratio: float) -> float:
        """Calculate optimal Kelly fraction"""
        if user_prob <= market_prob:
            return 0.0  # No positive edge, don't bet
        
        # Kelly formula: f = (bp - q) / b
        # where b = payout odds, p = win probability, q = lose probability
        b = payout_ratio
        p = user_prob
        q = 1 - user_prob
        
        kelly_fraction = (b * p - q) / b
        return max(0, kelly_fraction)
    
    def _calculate_expected_value(self, user_prob: float, stake: float, payout_ratio: float) -> float:
        """Calculate expected value of the bet"""
        win_amount = stake * payout_ratio
        lose_amount = stake
        
        expected_value = (user_prob * win_amount) - ((1 - user_prob) * lose_amount)
        return expected_value
    
    def _calculate_kelly_growth_rate(self, user_prob: float, kelly_fraction: float, payout_ratio: float) -> float:
        """Calculate expected logarithmic growth rate using Kelly"""
        if kelly_fraction <= 0:
            return 0.0
        
        # Growth rate = p * log(1 + f * b) + (1-p) * log(1 - f)
        p = user_prob
        f = kelly_fraction
        b = payout_ratio
        
        if f >= 1:  # Avoid log of negative numbers
            return -float('inf')
        
        growth_rate = p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)
        return growth_rate
    
    def _calculate_confidence_score(self, edge: float, kelly_fraction: float) -> float:
        """Calculate confidence score based on edge and Kelly fraction"""
        # Higher edge and reasonable Kelly fraction = higher confidence
        edge_score = min(abs(edge) * 10, 1.0)  # Normalize edge to 0-1
        kelly_score = min(kelly_fraction * 2, 1.0)  # Normalize Kelly to 0-1
        
        # Weighted average
        confidence = 0.7 * edge_score + 0.3 * kelly_score
        return min(confidence, 1.0)
    
    def _assess_risk_level(self, kelly_fraction: float, edge: float) -> str:
        """Assess risk level of the recommendation"""
        if kelly_fraction <= 0.05 or edge <= 0.02:
            return "low"
        elif kelly_fraction <= 0.15 or edge <= 0.10:
            return "medium"
        else:
            return "high"
    
    def analyze_portfolio_optimization(self, positions: list) -> Dict[str, Any]:
        """
        Analyze portfolio-level optimization for multiple positions
        
        Args:
            positions: List of position data with correlations
            
        Returns:
            Portfolio optimization recommendations
        """
        # This is a placeholder for future portfolio optimization
        # Would implement Modern Portfolio Theory concepts
        return {
            "status": "not_implemented",
            "message": "Portfolio optimization coming in future release"
        }
    
    def monte_carlo_simulation(self, analysis_input: GTOAnalysisInput, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to assess strategy performance
        
        Args:
            analysis_input: Input parameters
            num_simulations: Number of simulation runs
            
        Returns:
            Simulation results and statistics
        """
        try:
            results = []
            initial_bankroll = analysis_input.bankroll
            
            # Get basic analysis first
            analysis = self.analyze_binary_bet(analysis_input)
            stake = analysis.recommended_stake
            
            if stake <= 0:
                return {
                    "final_bankroll_mean": initial_bankroll,
                    "final_bankroll_std": 0,
                    "probability_of_profit": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0
                }
            
            for _ in range(num_simulations):
                bankroll = initial_bankroll
                outcome = np.random.random() < analysis_input.user_probability
                
                if outcome:
                    # Win
                    bankroll += stake * analysis_input.payout_ratio
                else:
                    # Lose
                    bankroll -= stake
                
                results.append(bankroll)
            
            results = np.array(results)
            
            return {
                "final_bankroll_mean": float(np.mean(results)),
                "final_bankroll_std": float(np.std(results)),
                "probability_of_profit": float(np.mean(results > initial_bankroll)),
                "max_drawdown": float(initial_bankroll - np.min(results)),
                "worst_case": float(np.min(results)),
                "best_case": float(np.max(results)),
                "percentile_5": float(np.percentile(results, 5)),
                "percentile_95": float(np.percentile(results, 95))
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {"error": str(e)}

# Singleton instance
gto_engine = GTOEngine()