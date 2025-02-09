import unittest
from askgloom.browser.browser import Browser

class TestBrowser(unittest.TestCase):
    def setUp(self):
        self.browser = Browser()

    def tearDown(self):
        if hasattr(self, 'browser'):
            self.browser.quit()

    def test_browser_initialization(self):
        """Test that browser initializes correctly"""
        self.assertIsNotNone(self.browser)
        self.assertIsNotNone(self.browser.driver)

    def test_navigate_to_url(self):
        """Test browser navigation to URL"""
        test_url = "https://example.com"
        self.browser.navigate(test_url)
        self.assertEqual(self.browser.current_url, test_url)

    def test_browser_profile_loading(self):
        """Test that browser profile loads correctly"""
        self.assertTrue(self.browser.profile_loaded)
        # Add more specific profile checks based on your implementation

if __name__ == '__main__':
    unittest.main()