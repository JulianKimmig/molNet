import json
import pickle
import traceback

from bs4 import BeautifulSoup
import requests


def get_all_chem_records():
    print("GET ALL RECORDS")
    url = "https://en.wikipedia.org/w/api.php?action=query&list=embeddedin&eititle=Template:Chembox&format=json&eilimit=500"
    resp = requests.get(url)
    data = json.loads(resp.text)
    full_data = data["query"]["embeddedin"]
    while "continue" in data:
        print(data["continue"]["eicontinue"].split("|")[1], end="\r")
        resp = requests.get(url + "&eicontinue=" + data["continue"]["eicontinue"])
        data = json.loads(resp.text)
        full_data.extend(data["query"]["embeddedin"])
    return full_data


def get_raw_wiki_data(full_data, book=None, max=None):
    all_data = {} if book is None else book
    ini_len = len(all_data)
    l = len(full_data)
    title = " " * 10
    if max is None:
        max = 2 * len(full_data)
    for i, page in enumerate(full_data):
        if page["ns"] in [1, 2, 3, 4, 5, 6, 10, 11]:
            continue  # skip namesapces https://en.wikipedia.org/wiki/Wikipedia:Namespace

        if len(all_data) - ini_len >= max:
            break
        pid = page["pageid"]
        if pid in all_data:
            continue
        try:
            url = "https://en.wikipedia.org/w/api.php?action=parse&format=json&pageid={}&prop=parsetree".format(
                pid
            )
            resp = requests.get(url)
            data = json.loads(resp.text)
            d = data["parse"]["parsetree"]["*"]
            # print(data.keys())
            ntitle = data["parse"]["title"]
            print(i, "/", l, " " * len(title), end="\r")
            print(i, "/", l, ntitle, end="\r")
            title = ntitle
            soup = BeautifulSoup(
                d,
            )
            for comment in soup.findAll("comment"):
                comment.decompose()
            for t in soup.findAll(text=True):
                text = str(t).strip()
                t.replaceWith(text.lower())

            box = soup.find("title", string="chembox")
            if box is None:
                for t in soup.findAll("title"):
                    if t.text == "chembox":
                        box = t
                        break

            if box is None:
                box = soup.find("title", string="infobox chemical")
                if box is None:
                    for t in soup.findAll("title"):
                        if t.text == "infobox chemical":
                            box = t
                            break
            box = box.parent

            def dispatch_template(tl):
                parts = tl.findAll("part", recursive=False)

                if len(parts) < 1:
                    return tl.find("title").text
                data = {}
                for part in parts:
                    name = part.findAll("name", recursive=False)
                    assert len(name) == 1
                    value = part.findAll("value", recursive=False)
                    assert len(value) == 1
                    name = name[0]
                    value = value[0]

                    if name.text == "":
                        key = name.attrs["index"]
                    else:
                        key = name.text
                    if key.startswith("section"):
                        try:
                            key = value.find("template").find("title").text
                        except:
                            pass
                    # print(key)
                    valchildren = value.findChildren(recursive=False)
                    if len(valchildren) == 0:
                        if value.text == "":
                            continue
                        data[key] = value.text
                    else:
                        vc_data = []
                        for vc in valchildren:
                            if vc.name == "template":
                                vc_data.append(dispatch_template(vc))
                            elif vc.name == "ext":
                                vc.decompose()
                                vc_data.append(value.text)
                            else:
                                raise NotImplementedError(vc)
                        data[key] = vc_data

                if list(data.keys()) == [str(i) for i in range(1, len(data) + 1)]:
                    data = [data[str(i)] for i in range(1, len(data) + 1)]
                return data

            data = dispatch_template(box)
            all_data[pid] = {"data": data, "title": title}
        except:
            traceback.print_exc()
            print(soup.findAll("title"))
            print(page)
            break
    return all_data


if __name__ == "__main__":

    try:
        with open("all_chem_records.pickle", "rb") as f:
            all_chem_records = pickle.load(f)
    except:
        with open("all_chem_records.pickle", "w+b") as f:
            pickle.dump(get_all_chem_records(), f)
        with open("all_chem_records.pickle", "rb") as f:
            all_chem_records = pickle.load(f)

    try:
        with open("all_chem_raw_wiki_data.pickle", "rb") as f:
            all_chem_raw_wiki_data = pickle.load(f)
    except:
        all_chem_raw_wiki_data = {}

    l = len(all_chem_raw_wiki_data)
    all_chem_raw_wiki_data = get_raw_wiki_data(
        all_chem_records, book=all_chem_raw_wiki_data, max=10
    )
    while len(all_chem_raw_wiki_data) > l:
        l = len(all_chem_raw_wiki_data)
        all_chem_raw_wiki_data = get_raw_wiki_data(
            all_chem_records, book=all_chem_raw_wiki_data, max=10
        )
        with open("all_chem_raw_wiki_data.pickle", "w+b") as f:
            pickle.dump(all_chem_raw_wiki_data, f)
        with open("all_chem_raw_wiki_data.pickle", "rb") as f:
            all_chem_raw_wiki_data = pickle.load(f)
