# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


import datetime
import unittest

from asyncpg._testbase import ConnectedTestCase


class TestTimetzSubminuteOffset(ConnectedTestCase):

    async def test_timetz_keeps_subminute_offset(self):
        cases = [
            (
                '12:00:00+00:00:30',
                datetime.time(
                    12, 0, 0,
                    tzinfo=datetime.timezone(
                        datetime.timedelta(seconds=30))),
            ),
            (
                '12:00:00-00:00:30',
                datetime.time(
                    12, 0, 0,
                    tzinfo=datetime.timezone(
                        datetime.timedelta(seconds=-30))),
            ),
            (
                '12:00:00+05:30:15',
                datetime.time(
                    12, 0, 0,
                    tzinfo=datetime.timezone(
                        datetime.timedelta(
                            hours=5, minutes=30, seconds=15))),
            ),
            (
                '00:01:00-03:30',
                datetime.time(
                    0, 1, 0,
                    tzinfo=datetime.timezone(
                        datetime.timedelta(hours=-3, minutes=-30))),
            ),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                got = await self.con.fetchval(
                    'SELECT $1::text::timetz', text)
                self.assertEqual(got, expected)


if __name__ == '__main__':
    unittest.main()
