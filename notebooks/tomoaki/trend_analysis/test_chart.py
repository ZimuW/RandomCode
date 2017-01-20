#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import unittest
import json
from mock import MagicMock, patch

from flask import url_for
from flask.ext.testing import TestCase
import pandas as pd

import models
from models import Strategy, User, func
from database import db_session

import www.views.chart as module


class TestChart(TestCase):

    @classmethod
    def setUpClass(cls):
        models.cleanup()
        user = User(
            name='test',
            email='test@alpacadb.com',
            password='abc',
            timezone='Asia/Tokyo',
            confirmed_at=func.now())
        db_session.add(user)
        db_session.commit()

        st = Strategy(name="Test Strategy", symbol="EURUSD", timeframe="5Min",
                      user_id=user.id, indicators=[])
        db_session.add(st)
        db_session.commit()

    @classmethod
    def tearDownClass(cls):
        db_session.remove()
        models.cleanup()
        from www.manage import create_test_users
        create_test_users(silent=True)

    def tearDown(self):
        reload(module)

    def _login(self, email="yuki@alpacadb.com"):
        self.client.post(url_for('account.signin'), data=dict(
            email=email, password='yuki',
        ))

    def _logout(self):
        self.client.post(url_for('account.signout'))

    def create_app(self):
        from www.manage import create_app

        return create_app("settings.TestConfig")

    def api_call(self, method, params):
        return self.client.post(
            "/admin/api",
            content_type='application/json',
            data=json.dumps({
                'jsonrpc': '2.0',
                'method': method,
                'params': params,
                'id': 1}))

    def test_backtest_chart(self):

        import numpy as np
        bt = MagicMock()
        bt.minute_total = pd.Series(
            data=np.arange(11),
            index=pd.date_range(
                '2015-12-22 11:00Z',
                '2015-12-22 11:10Z',
                freq='1Min'))

        with patch.object(module, 'db_session') as db_session:

            db_session.query().get.return_value = bt

            self._login()
            rv = self.client.get(
                url_for(
                    'chart.backtest',
                    backtest_id=0))
            self._logout()
            self.assertEqual(rv.status_code, 200)

    def test_livetest_chart(self):

        import numpy as np
        with patch.object(
                module,
                'get_livetest_performance') as get_livetest_performance:

            get_livetest_performance.return_value = (None, pd.Series(
                data=np.arange(11),
                index=pd.date_range(
                    '2015-12-22 11:00Z',
                    '2015-12-22 11:10Z',
                    freq='1Min')))

            rv = self.client.get(
                url_for(
                    'chart.livetest',
                    strategy_id=0))
            self.assertEqual(rv.status_code, 200)

            # empty
            get_livetest_performance.return_value = (None, pd.Series(
                data=[],
                index=[]))

            rv = self.client.get(
                url_for(
                    'chart.livetest',
                    strategy_id=0))
            self.assertEqual(rv.status_code, 200)

    def test_chart_thumbnail(self):
        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="EURUSD",
                timeframe="1Min",
                frm="201506020000",
                to="201506020100",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 200)

        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="USDJPY",
                timeframe="5Min",
                frm="201506020000",
                to="201506030000",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 200)

        # unavailable symbol
        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="USDCHF--",
                timeframe="15Min",
                frm="201506020000",
                to="201506030000",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 404)

    def test_chart_thumbnail_invalid_frm_to(self):
        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="EURUSD",
                timeframe="15Min",
                frm="2015-06-020000",
                to="20150603-0000",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 400)

    @patch.object(module, 'get_range')
    @patch.object(module, 'requests')
    @patch.object(module, 'cacher')
    def test_chart_thumbnail_chart_server_returns_404(
            self, cacher, requests, get_range):
        cacher.get.return_value = None
        response = MagicMock()
        response.status_code = 404
        requests.post.return_value = response
        get_range.return_value = {}
        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="USDCHF",
                timeframe="15Min",
                frm="201506020000",
                to="201506030000",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 400)

    @patch.object(module, 'get_range')
    @patch.object(module, 'requests')
    @patch.object(module, 'cacher')
    def test_chart_thumbnail_chart_server_returns_invalid_html(
            self, cacher, requests, get_range):
        cacher.get.return_value = None
        response = MagicMock()
        response.status_code = 200
        response.text = "<svg>test</svg>"
        requests.post.return_value = response
        get_range.return_value = {}
        rv = self.client.get(
            url_for(
                'chart.thumbnail',
                symbol="USDCHF",
                timeframe="15Min",
                frm="201506020000",
                to="201506030000",
                draw_frm=1
            ))
        self.assertEqual(rv.status_code, 400)

    def test_chart_full(self):
        rv = self.client.get(
            url_for(
                'chart.full',
                symbol="EURUSD",
                timeframe="1Min",
                frm="201506020000",
                to="201506020100",
                type="svg"
            ))
        self.assertEqual(rv.status_code, 200)

        rv = self.client.get(
            url_for(
                'chart.full',
                symbol="USDJPY",
                timeframe="5Min",
                frm="201506020000",
                to="201506030000",
                type="svg"
            ))
        self.assertEqual(rv.status_code, 200)

        # unavailable symbol
        rv = self.client.get(
            url_for(
                'chart.full',
                symbol="USDCHF--",
                timeframe="15Min",
                frm="201506020000",
                to="201506030000",
                type="png"
            ))
        self.assertEqual(rv.status_code, 404)

if __name__ == '__main__':
    unittest.main()
