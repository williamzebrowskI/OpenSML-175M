"""Offline tests only; never access real SMC or launch sensor processes."""
import ctypes
import math
import struct
import unittest
from unittest.mock import patch

from scripts import rack_sensors as sensors
from scripts.rack_dashboard import clean_hardware, network_rates, Health


class SensorsTests(unittest.TestCase):
    def test_zero_is_measured_and_invalid_values_are_unknown(self):
        data = sensors.parse_macmon({'gpu_usage': [0, 0], 'gpu_power': 0,
            'ecpu_usage': [1200, .2], 'pcpu_usage': [1200, math.nan],
            'sys_power': math.inf, 'temp': {'gpu_temp_avg': 1.745}})
        self.assertEqual(data['gpu_load'], 0)
        self.assertEqual(data['gpu_power'], 0)
        self.assertEqual(data['ecpu_load'], 20)
        self.assertIsNone(data['pcpu_load'])
        self.assertIsNone(data['sys_power'])
        self.assertNotIn('gpu_temp', data)
        self.assertEqual(sensors.parse_macmon([]), {})

    def test_allowlist(self):
        clean = clean_hardware({'gpu_load': 101, 'cpu_temp': -20, 'gpu_temp': 1.7,
            'sys_power': '300', 'ram_usage': 100, 'ram_total': 90,
            'private': 'secret', 'fan_rpm': [1010, -4, 3000],
            'network_counters': {'en4': {'rx': 0, 'tx': 10}, 'invalid': {'rx': 2, 'tx': 3},
                                 'en5': {'rx': math.inf, 'tx': 1}}})
        self.assertIsNone(clean['gpu_load'])
        self.assertIsNone(clean['cpu_temp'])
        self.assertIsNone(clean['gpu_temp'])
        self.assertIsNone(clean['ram_usage'])
        self.assertNotIn('private', clean)
        self.assertEqual(clean['fan_rpm'], [1010])
        self.assertEqual(set(clean['network_counters']), {'en4'})

    def test_smc_layout_and_decoders(self):
        self.assertEqual(ctypes.sizeof(sensors.KeyData), 80)
        self.assertEqual(sensors.KeyData.bytes.offset, 48)
        self.assertEqual(sensors.decode_smc(b'flt ', struct.pack('<f', 43.5)), 43.5)
        self.assertEqual(sensors.decode_smc(b'sp78', b'\x2a\x80'), 42.5)
        self.assertEqual(sensors.decode_smc(b'fpe2', (4000).to_bytes(2, 'big')), 1000)
        self.assertIsNone(sensors.decode_smc(b'????', b'abcd'))

    def test_netstat_reads_link_rows_once(self):
        raw = '''Name Mtu Network Address Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll
en4 1500 <Link#1> aa:bb 100 0 10000 100 0 20000 0
en4 1500 10.0.0.0/30 10.0.0.1 100 - 10000 100 - 20000 -
en1* 1500 <Link#2> aa:bb 0 0 0 0 0 0 0
lo0 16384 <Link#3> 40 0 1000 40 0 1000 0
en2 malformed
'''
        self.assertEqual(sensors.parse_network(raw), {'en4': {'rx': 10000, 'tx': 20000}, 'en1': {'rx': 0, 'tx': 0}})

    def test_network_rates_missing_reset_and_gap(self):
        first = {'network_time': 100, 'network_counters': {'en4': {'rx': 100, 'tx': 200}, 'en5': {'rx': 100, 'tx': 200}}}
        second = {'network_time': 110, 'network_counters': {'en4': {'rx': 200, 'tx': 400}, 'en5': {'rx': 200, 'tx': 400}}}
        rate = network_rates(first, second, ['en4', 'en5'])
        self.assertEqual(rate['tb_rx_Bps'], 20)
        self.assertEqual(rate['tb_tx_Bps'], 40)
        self.assertIsNone(rate['wifi_rx_Bps'])
        self.assertIsNone(network_rates({}, second, ['en4'])['tb_rx_Bps'])
        self.assertIsNone(network_rates(second, first, ['en4'])['tb_rx_Bps'])
        second['network_time'] = 200
        self.assertIsNone(network_rates(first, second, ['en4'])['tb_rx_Bps'])
        second['network_time'] = 110
        second['network_counters']['en4']['rx'] = 10
        self.assertIsNone(network_rates(first, second, ['en4'])['tb_rx_Bps'])

    def test_missing_macmon_preserves_other_sensors(self):
        with patch.object(sensors.subprocess, 'check_output', side_effect=FileNotFoundError), \
             patch.object(sensors, 'smc_sensors', return_value={'cpu_temp': 40}):
            result = sensors.sample_hardware('Apple M4 Max')
        self.assertEqual(result['cpu_temp'], 40)
        self.assertNotIn('gpu_load', result)

    def test_sampler_history_is_bounded_and_failures_do_not_reuse_load(self):
        health = Health(enabled=False, probe=lambda rank: {'rank': rank, 'checked_at': 100,
            'hardware': {'gpu_load': 20}, 'reachable': True})
        for _ in range(70):
            health._sample(0)
        self.assertEqual(len(health.nodes[0]['hardware']['gpu_history']), 60)
        health.probe = lambda rank: (_ for _ in ()).throw(OSError())
        health._sample(0)
        self.assertIsNone(health.nodes[0]['reachable'])
        self.assertNotIn('gpu_load', health.nodes[0]['hardware'])
        self.assertIsNone(health.nodes[0]['hardware']['gpu_history'][-1]['gpu'])
        health.close()


if __name__ == '__main__':
    unittest.main()
