import unittest
import xmlrunner

class TextClassifierTest(unittest.TestCase):

    def test_upper(self):
      self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))