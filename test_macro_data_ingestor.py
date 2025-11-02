"""
Tests for MacroDataIngestor.

Tests data fetching, caching, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from active_trader_llm.data_ingestion.macro_data_ingestor import MacroDataIngestor
from active_trader_llm.feature_engineering.models import MacroSnapshot


class TestMacroDataIngestor:
    """Test suite for MacroDataIngestor"""

    def test_initialization(self):
        """Test ingestor initializes correctly"""
        ingestor = MacroDataIngestor(cache_duration_seconds=600)

        assert ingestor.cache_duration == timedelta(seconds=600)
        assert ingestor._cache is None
        assert ingestor._cache_timestamp is None

    def test_cache_validity_check(self):
        """Test cache validity checking"""
        ingestor = MacroDataIngestor(cache_duration_seconds=300)

        # No cache - should be invalid
        assert not ingestor._is_cache_valid()

        # Add cache with current timestamp
        ingestor._cache = MacroSnapshot(timestamp=datetime.now().isoformat())
        ingestor._cache_timestamp = datetime.now()

        # Should be valid
        assert ingestor._is_cache_valid()

        # Old cache - should be invalid
        ingestor._cache_timestamp = datetime.now() - timedelta(seconds=400)
        assert not ingestor._is_cache_valid()

    @patch('yfinance.Ticker')
    def test_fetch_single_value_success(self, mock_ticker_class):
        """Test successful single value fetch"""
        ingestor = MacroDataIngestor()

        # Mock yfinance response
        mock_ticker = Mock()
        mock_hist = pd.DataFrame({'Close': [17.5]})
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        value = ingestor._fetch_single_value('^VIX', 'VIX')

        assert value == 17.5
        mock_ticker.history.assert_called_once_with(period="1d")

    @patch('yfinance.Ticker')
    def test_fetch_single_value_empty_data(self, mock_ticker_class):
        """Test handling of empty data"""
        ingestor = MacroDataIngestor()

        # Mock empty response
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        value = ingestor._fetch_single_value('^VIX', 'VIX')

        assert value is None

    @patch('yfinance.Ticker')
    def test_fetch_single_value_exception(self, mock_ticker_class):
        """Test handling of fetch exceptions"""
        ingestor = MacroDataIngestor()

        # Mock exception
        mock_ticker_class.side_effect = Exception("Network error")

        value = ingestor._fetch_single_value('^VIX', 'VIX')

        assert value is None

    @patch.object(MacroDataIngestor, '_fetch_single_value')
    def test_fetch_volatility_indices(self, mock_fetch):
        """Test volatility indices fetching"""
        ingestor = MacroDataIngestor()

        # Mock successful fetches
        mock_fetch.side_effect = [17.5, 21.2, 105.3]

        result = ingestor.fetch_volatility_indices()

        assert result['vix'] == 17.5
        assert result['vxn'] == 21.2
        assert result['move_index'] == 105.3
        assert mock_fetch.call_count == 3

    @patch.object(MacroDataIngestor, '_fetch_single_value')
    def test_fetch_treasury_yields(self, mock_fetch):
        """Test treasury yields fetching and spread calculation"""
        ingestor = MacroDataIngestor()

        # Mock yields
        mock_fetch.side_effect = [4.10, 3.72, 4.67]

        result = ingestor.fetch_treasury_yields()

        assert result['treasury_10y'] == 4.10
        assert result['treasury_2y'] == 3.72
        assert result['treasury_30y'] == 4.67
        # Check spread calculation
        assert result['yield_curve_spread'] == pytest.approx(0.38, abs=0.01)

    @patch.object(MacroDataIngestor, '_fetch_single_value')
    def test_fetch_treasury_yields_missing_data(self, mock_fetch):
        """Test treasury yields with missing data"""
        ingestor = MacroDataIngestor()

        # Mock partial data
        mock_fetch.side_effect = [None, 3.72, 4.67]

        result = ingestor.fetch_treasury_yields()

        assert result['treasury_10y'] is None
        assert result['treasury_2y'] == 3.72
        # Spread should be None if 10Y missing
        assert result['yield_curve_spread'] is None

    @patch.object(MacroDataIngestor, '_fetch_single_value')
    def test_fetch_commodities(self, mock_fetch):
        """Test commodities fetching"""
        ingestor = MacroDataIngestor()

        mock_fetch.side_effect = [3982.20, 60.98]

        result = ingestor.fetch_commodities()

        assert result['gold_price'] == 3982.20
        assert result['oil_price'] == 60.98

    @patch.object(MacroDataIngestor, '_fetch_single_value')
    def test_fetch_currency(self, mock_fetch):
        """Test currency fetching"""
        ingestor = MacroDataIngestor()

        mock_fetch.return_value = 99.80

        result = ingestor.fetch_currency()

        assert result['dollar_index'] == 99.80

    def test_fetch_nyse_breadth(self):
        """Test NYSE breadth fetching"""
        ingestor = MacroDataIngestor()

        # Mock the breadth ingestor
        with patch.object(ingestor.breadth_ingestor, 'fetch_latest_breadth') as mock_fetch:
            mock_fetch.return_value = {
                'nyse_advancing': 1850,
                'nyse_declining': 1120,
                'nyse_unchanged': 130,
                'nyse_advancing_volume': 450000000,
                'nyse_declining_volume': 280000000,
                'nyse_new_highs': 85,
                'nyse_new_lows': 42,
                'advance_decline_ratio': 1.65,
                'up_volume_ratio': 0.616
            }

            result = ingestor.fetch_nyse_breadth()

            assert result['nyse_advancing'] == 1850
            assert result['nyse_declining'] == 1120
            assert result['advance_decline_ratio'] == pytest.approx(1.65)
            assert result['up_volume_ratio'] == pytest.approx(0.616)
            mock_fetch.assert_called_once_with(use_cache=True)

    @patch.object(MacroDataIngestor, 'fetch_volatility_indices')
    @patch.object(MacroDataIngestor, 'fetch_treasury_yields')
    @patch.object(MacroDataIngestor, 'fetch_commodities')
    @patch.object(MacroDataIngestor, 'fetch_currency')
    @patch.object(MacroDataIngestor, 'fetch_nyse_breadth')
    def test_fetch_all_success(self, mock_breadth, mock_currency, mock_commodities, mock_yields, mock_volatility):
        """Test fetch_all with successful data"""
        ingestor = MacroDataIngestor()

        # Mock all fetchers
        mock_volatility.return_value = {'vix': 17.5, 'vxn': 21.2, 'move_index': 105.3}
        mock_yields.return_value = {
            'treasury_10y': 4.10,
            'treasury_2y': 3.72,
            'treasury_30y': 4.67,
            'yield_curve_spread': 0.38
        }
        mock_commodities.return_value = {'gold_price': 3982.20, 'oil_price': 60.98}
        mock_currency.return_value = {'dollar_index': 99.80}
        mock_breadth.return_value = {
            'nyse_advancing': 1850,
            'nyse_declining': 1120,
            'nyse_unchanged': 130,
            'nyse_advancing_volume': 450000000,
            'nyse_declining_volume': 280000000,
            'nyse_new_highs': 85,
            'nyse_new_lows': 42,
            'advance_decline_ratio': 1.65,
            'up_volume_ratio': 0.616
        }

        snapshot = ingestor.fetch_all(use_cache=False)

        assert isinstance(snapshot, MacroSnapshot)
        assert snapshot.vix == 17.5
        assert snapshot.vxn == 21.2
        assert snapshot.treasury_10y == 4.10
        assert snapshot.yield_curve_spread == pytest.approx(0.38)
        assert snapshot.gold_price == 3982.20
        assert snapshot.dollar_index == 99.80
        # Test breadth fields
        assert snapshot.nyse_advancing == 1850
        assert snapshot.nyse_declining == 1120
        assert snapshot.advance_decline_ratio == pytest.approx(1.65)
        assert snapshot.up_volume_ratio == pytest.approx(0.616)

    @patch.object(MacroDataIngestor, 'fetch_volatility_indices')
    @patch.object(MacroDataIngestor, 'fetch_treasury_yields')
    @patch.object(MacroDataIngestor, 'fetch_commodities')
    @patch.object(MacroDataIngestor, 'fetch_currency')
    @patch.object(MacroDataIngestor, 'fetch_nyse_breadth')
    def test_fetch_all_partial_data(self, mock_breadth, mock_currency, mock_commodities, mock_yields, mock_volatility):
        """Test fetch_all with partial data"""
        ingestor = MacroDataIngestor()

        # Mock partial data (some None values)
        mock_volatility.return_value = {'vix': 17.5, 'vxn': None, 'move_index': None}
        mock_yields.return_value = {
            'treasury_10y': 4.10,
            'treasury_2y': None,
            'treasury_30y': None,
            'yield_curve_spread': None
        }
        mock_commodities.return_value = {'gold_price': None, 'oil_price': 60.98}
        mock_currency.return_value = {'dollar_index': 99.80}
        mock_breadth.return_value = {
            'nyse_advancing': 1850,
            'nyse_declining': None,
            'nyse_unchanged': None,
            'nyse_advancing_volume': None,
            'nyse_declining_volume': None,
            'nyse_new_highs': None,
            'nyse_new_lows': None,
            'advance_decline_ratio': None,
            'up_volume_ratio': None
        }

        snapshot = ingestor.fetch_all(use_cache=False)

        # Should still create snapshot with available data
        assert snapshot.vix == 17.5
        assert snapshot.vxn is None
        assert snapshot.treasury_10y == 4.10
        assert snapshot.treasury_2y is None
        assert snapshot.oil_price == 60.98
        assert snapshot.dollar_index == 99.80
        assert snapshot.nyse_advancing == 1850
        assert snapshot.nyse_declining is None

    @patch.object(MacroDataIngestor, 'fetch_volatility_indices')
    @patch.object(MacroDataIngestor, 'fetch_treasury_yields')
    @patch.object(MacroDataIngestor, 'fetch_commodities')
    @patch.object(MacroDataIngestor, 'fetch_currency')
    @patch.object(MacroDataIngestor, 'fetch_nyse_breadth')
    def test_fetch_all_caching(self, mock_breadth, mock_currency, mock_commodities, mock_yields, mock_volatility):
        """Test fetch_all caching behavior"""
        ingestor = MacroDataIngestor(cache_duration_seconds=300)

        # Mock data
        mock_volatility.return_value = {'vix': 17.5, 'vxn': 21.2, 'move_index': 105.3}
        mock_yields.return_value = {
            'treasury_10y': 4.10,
            'treasury_2y': 3.72,
            'treasury_30y': 4.67,
            'yield_curve_spread': 0.38
        }
        mock_commodities.return_value = {'gold_price': 3982.20, 'oil_price': 60.98}
        mock_currency.return_value = {'dollar_index': 99.80}
        mock_breadth.return_value = {
            'nyse_advancing': 1850,
            'nyse_declining': 1120,
            'nyse_unchanged': 130,
            'nyse_advancing_volume': 450000000,
            'nyse_declining_volume': 280000000,
            'nyse_new_highs': 85,
            'nyse_new_lows': 42,
            'advance_decline_ratio': 1.65,
            'up_volume_ratio': 0.616
        }

        # First fetch - should call all fetchers
        snapshot1 = ingestor.fetch_all(use_cache=True)
        assert mock_volatility.call_count == 1

        # Second fetch - should use cache
        snapshot2 = ingestor.fetch_all(use_cache=True)
        assert mock_volatility.call_count == 1  # Not called again
        assert snapshot2 == snapshot1

    def test_timestamp_format(self):
        """Test timestamp is in ISO format"""
        ingestor = MacroDataIngestor()

        breadth_none = {
            'nyse_advancing': None, 'nyse_declining': None, 'nyse_unchanged': None,
            'nyse_advancing_volume': None, 'nyse_declining_volume': None,
            'nyse_new_highs': None, 'nyse_new_lows': None,
            'advance_decline_ratio': None, 'up_volume_ratio': None
        }

        with patch.object(ingestor, 'fetch_volatility_indices', return_value={'vix': None, 'vxn': None, 'move_index': None}):
            with patch.object(ingestor, 'fetch_treasury_yields', return_value={'treasury_10y': None, 'treasury_2y': None, 'treasury_30y': None, 'yield_curve_spread': None}):
                with patch.object(ingestor, 'fetch_commodities', return_value={'gold_price': None, 'oil_price': None}):
                    with patch.object(ingestor, 'fetch_currency', return_value={'dollar_index': None}):
                        with patch.object(ingestor, 'fetch_nyse_breadth', return_value=breadth_none):
                            snapshot = ingestor.fetch_all(use_cache=False)

                            # Should be valid ISO timestamp
                            datetime.fromisoformat(snapshot.timestamp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
