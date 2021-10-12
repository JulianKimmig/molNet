import pickle
from pprint import pprint

from bs4 import BeautifulSoup
import requests


def request_soup(link, session=None):
    if session is None:
        session = requests.Session()
    r = session.get(link)
    soup: BeautifulSoup = BeautifulSoup(r.content, "html.parser")
    return soup, session


BASEURL = "https://ochem.eu/"

soup, session = request_soup(BASEURL + "alerts/list.do")


login_page_link = soup.find(
    "iframe",
    src=lambda value: value and value.startswith(BASEURL + "login/show.do;jsessionid"),
)["src"]

soup, session = request_soup(login_page_link, session=session)


anonym_login_link = (
    BASEURL
    + soup.find(
        "a", href=lambda value: value and value.startswith("login/login.do?anonymous=1")
    )["href"]
)


soup, session = request_soup(anonym_login_link, session=session)


accept_licence_link = BASEURL + soup.find(
    "a",
    href=lambda value: value
    and value.startswith("login/acceptLicenseAgreement.do?")
    and value.endswith("accepted=1"),
)["href"]

soup, session = request_soup(accept_licence_link, session=session)


# soup,session = request_soup(BASEURL+"alerts/list.do",session=session)


def get_alert_page(p, s):
    return s.post(
        BASEURL + "alerts/list.do",
        data={
            "out": "json",
            "pagenum": str(p),
            "render-mode": "popup",
        },
    ).json()


data = []

cp = 1
page = get_alert_page(cp, session)
data.extend(page["list"]["substructure-alert"])
while int(page["list"]["lastResult"]) < int(page["list"]["size"]):
    cp += 1
    page = get_alert_page(cp, session)
    print(page["list"]["pageNum"])
    data.extend(page["list"]["substructure-alert"])

with open("ochem_alerts.pickle", "w+b") as f:
    pickle.dump(data, f)
