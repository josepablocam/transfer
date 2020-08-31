# crawl kaggle for a given data set and download all python source
from argparse import ArgumentParser
import logging
import os
import re
from urllib.parse import urljoin
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tqdm
import wget

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
log = logging.getLogger(__name__)


def parse_source(page_source):
    return BeautifulSoup(page_source, 'html5lib')


def make_browser():
    browser = webdriver.Firefox()
    # make sure we don't get mobile websites by setting size
    browser.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    return browser


class KaggleCrawler(object):
    def __init__(
        self,
        kernels_url,
        output_dir,
        language='Python',
        sort_by='Most Votes'
    ):
        self.kernels_url = kernels_url
        # specify crawl kernel parameters
        self.language = language
        self.sort_by = sort_by
        self.output_dir = output_dir
        # keep track of what we've downloaded/tried to
        self.downloaded_links = set([])
        self.failed_links = set([])
        self.browser_references = []

        self.browser = make_browser()
        self.browser_references.append(self.browser)
        self.browser.get(self.kernels_url)
        log.info('Created crawler for {}'.format(kernels_url))

    @staticmethod
    def get_base_url():
        return "https://www.kaggle.com/"

    @staticmethod
    def get_dropdown_constants():
        menus = {
            'language':
            set(['Languages', 'Julia', 'R', 'Python', 'SQLite']),
            'sort':
            set([
                'Hotness', 'Most Votes', 'Most Comments', 'Recently Created',
                'Recently Run'
            ]),
        }
        return menus

    @staticmethod
    def _get_drop_down_options(which):
        """ Possible options in particular drop down menu """
        drop_down_constants = KaggleCrawler.get_dropdown_constants()
        if not which in drop_down_constants.keys():
            raise ValueError(
                "Drop down menu must be one of %s: %s" %
                (drop_down_constants.keys(), which)
            )
        return drop_down_constants[which]

    def _get_drop_down(self, which):
        """ Find dropdown in website by identifying one of the options """
        # retrieve all possible drop downs
        dropdowns = self.browser.find_elements_by_xpath(
            "//div[contains(@class, 'Select-value')]"
        )
        options = self._get_drop_down_options(which)
        for d in dropdowns:
            if d.text in options:
                return d
        raise Exception("No matching options in drop downs")

    def _get_drop_down_list_elem(self, dropdown):
        dropdown.click()
        elems = self.browser.find_elements_by_xpath(
            "//div[contains(@class, 'Select-input')]"
        )
        expanded = []
        for elem in elems:
            if elem.get_attribute('aria-expanded') == 'true':
                expanded.append(elem)
        if len(expanded) > 1:
            print(
                'Warning: multiple drop down menus are expanded, returning first'
            )
        return expanded[0]

    def _select_drop_down(self, which, option):
        """ Scroll a given drop down to a particular option. True if scroll succesfull false otherwise """
        drop_down = self._get_drop_down(which)
        options = self._get_drop_down_options(which)
        if not option.upper() in map(lambda x: x.upper(), options):
            raise ValueError(
                "No such option for menu %s: %s" % (which, option)
            )

        # we need to get the dropdown list from the Select-value element
        dropdown_list = self._get_drop_down_list_elem(drop_down)

        while True:
            if drop_down.text.upper() == option.upper():
                return True
            else:
                if dropdown_list.get_attribute('aria-expanded') == 'false':
                    drop_down.click()
                    time.sleep(2)
                dropdown_list.send_keys(Keys.ARROW_DOWN)
                dropdown_list.send_keys(Keys.ENTER)
                time.sleep(2)

    def _select_language(self, lang):
        """ Filter kernels to a given language """
        log.info('Filtering kernels to {}'.format(lang))
        assert self._select_drop_down('language', lang)

    def _sort_by(self, criteria):
        """ Sort kernels by given criteria """
        log.info('Sorting kernels by {}')
        assert self._select_drop_down('sort', criteria)

    def _get_height(self):
        """ Get height of scroll """
        return self.browser.execute_script("return document.body.scrollHeight")

    def _scroll_pg_down(self):
        """ Raw scroll, returns after executing, even if loading is not done """
        # scroll by entire page
        self.browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        # and scroll to see loading message
        self.browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight - 10);"
        )

    def _blocked_scroll_down(self, delay):
        """ Blocked scroll down. Checks that message for loading more kernels appears and disappears """
        try:
            # wait until kernel loading message appears/disappears
            wait = WebDriverWait(self.browser, delay)
            kernels_loading_msg = (
                By.XPATH, "//*[. = 'Loading more kernels...']"
            )
            # raw full page scroll
            self._scroll_pg_down()
            # wait until visible
            wait.until(EC.visibility_of_element_located(kernels_loading_msg))
            wait.until_not(
                EC.visibility_of_element_located(kernels_loading_msg)
            )
        except TimeoutException:
            log.warn('Timed out on scroll')

    def _infinite_scroll_kernels(self, n_scrolls=None, batch_size=10):
        """
        Scroll infinite scroll of kernels down by n_scrolls if provided or until height doesn't change.
        Prints count of scrolls every batch_size. Returns True if scrolled down
        """
        # TODO: could change this to check for No more kernels message instead,  might be cleaner
        if n_scrolls is not None and n_scrolls <= 0:
            raise ValueError("Must scroll at least once: %d" % n_scrolls)
        curr = 0
        while n_scrolls is None or curr < n_scrolls:
            if curr % batch_size == 0:
                print("Scroll: %d" % curr)
            current_height = self._get_height()
            self._scroll_pg_down()
            time.sleep(10)
            new_height = self._get_height()
            if current_height == new_height:
                log.info('Window height unchanged, done scrolling')
                return False
            curr += 1
        return True

    def _get_new_kernel_links(self):
        """
      Only returns new links (ignores previously downloaded).
      """
        parsed_src = parse_source(self.browser.page_source)
        raw_links = parsed_src.findAll('a', {'class': 'block-link__anchor'})
        raw_links = set([
            urljoin(self.get_base_url(), link['href']) for link in raw_links
        ])
        # remove links we've already downloaded or tried to and failed
        new_links = raw_links - self.downloaded_links - self.failed_links
        return list(new_links)

    @staticmethod
    def _get_wget_link(kernel_browser):
        code_tab = kernel_browser.find_element_by_xpath(
            '//span[contains(text(), "Code")]'
        )
        code_tab.click()
        pane = kernel_browser.find_element_by_class_name(
            'script-code-pane__download'
        )
        link = pane.get_attribute('href')
        return link

    def _download_source_code(self, kernel_browser, kernel_link, filename):
        log.info('Downloading code for {}'.format(kernel_link))
        kernel_browser.get(kernel_link)
        wget_link = self._get_wget_link(kernel_browser)
        download_result = wget.download(wget_link, out=self.output_dir)
        # file extension
        ext = download_result.split('.')[-1]
        new_path = os.path.join(self.output_dir, filename + '.' + ext)
        os.rename(download_result, new_path)

    def run(self, n_scrolls=10, batch_size=5, scroll_failure_budget=10):
        """
        Download scripts for a given data set
        """
        # infinite scrolling params
        n_scrolls = n_scrolls
        batch_size = batch_size
        # we can run out of kernels to scroll down to, so just stop after failing a lot
        scroll_failure_budget = scroll_failure_budget

        # browse to initial kernels site
        self.browser.get(self.kernels_url)
        self._select_language(self.language)
        self._sort_by(self.sort_by)

        # have separate browser to download individual kernels
        # we want to avoid losing infinite scroll on main crawler browser
        download_browser = make_browser()
        self.browser_references.append(download_browser)
        ct_downloaded = 0

        sources_file = open(os.path.join(self.output_dir, 'sources.txt'), 'w')
        log.info(
            'Writing sources for downloaded files to {}'.format(sources_file)
        )

        while scroll_failure_budget > 0:
            # scroll to find new links
            scrolled = self._infinite_scroll_kernels(
                n_scrolls=n_scrolls, batch_size=batch_size
            )
            scroll_failure_budget -= 1 if not scrolled else 0
            new_links = self._get_new_kernel_links()
            for link in tqdm.tqdm(new_links):
                # use same browser, to avoid openning a ton of firefox windows
                try:
                    filename = 'kernel_%d' % ct_downloaded
                    self._download_source_code(
                        download_browser, link, filename
                    )
                    self.downloaded_links.add(link)
                    sources_file.write('{} = {}\n'.format(filename, link))
                    ct_downloaded += 1
                except Exception as err:
                    log.exception('Failed to download {}'.format(link))
                    # keep track of failures to avoid repeating
                    self.failed_links.add(link)
            print("Total Downloaded: %d" % ct_downloaded)

        sources_file.close()

    def cleanup(self):
        for browser in self.browser_references:
            browser.quit()


def main(args):
    crawler = KaggleCrawler(args.kernels_url, args.output_dir)
    try:
        crawler.run()
    except Exception as err:
        import pdb
        pdb.post_mortem()
    finally:
        crawler.cleanup()


if __name__ == '__main__':
    parser = ArgumentParser(
        'Download all kernels from a given dataset by starting at kernels URL'
    )
    parser.add_argument(
        'kernels_url',
        type=str,
        help='URL to kernels homepage for given data set'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to existing directory to save kernels'
    )
    args = parser.parse_args()
    main(args)
