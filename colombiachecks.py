
import os
import re
import requests
import csv
from bs4 import BeautifulSoup


class Extractor:
    """

    """

    def __init__(self, url):
        """

        """

        self.url = url
        self.key = self.__define_key()
        self.soup = self.__create_soup()
        self.__extract()

    def __create_soup(self):
        """

        """

        try:
            req = requests.get(self.url)
        except Exception as e:
            print('Error:')
            print(e, "\n")

        if req.status_code == 200:

            soup_object = BeautifulSoup(req.text, "lxml")
            return soup_object

    def __extract(self):
        """

        """

        if self.key == 'colombiacheck':
            self.__colombiacheck_extractor()

    def __define_key(self):
        """

        """

        key_temp = re.match(r"https://(.*)\.com", self.url)

        if 'www.' in key_temp.group(1):

            key = key_temp.group(1).replace('www.', '')
            return key
        else:
            return key_temp.group(1)

    def __colombiacheck_extractor(self):
        """

        """
        checks_links = []
        div_checks = self.soup.find("div", {"id": "bloqueImagenes"})

        for i in div_checks.find_all("div", {"class": "Chequeo"}):

            link = i.find('a')
            try:
                checks_links.append(link.get("href"))
            except Exception as e:
                print("link append error: ")
                print(e)

        checks_url = ["https://colombiacheck.com" + x for x in checks_links]

        checks_content = []

        for link in checks_url:

            try:
                check = requests.get(link)

            except Exception as e:
                print("Error: ")
                print(e)

            if check.status_code == 200:
                check_soup = BeautifulSoup(check.text, "lxml")

            # Value of Check
            check_p = check_soup.find("div", {"class": "Portada-bandera"})

            check_value = check_p.find(
                "p", {"class": "Portada-bandera-text"}).get_text().strip()

            check_content = check_soup.find(
                "div", {"class": "text-articulos"})

            # Check Date
            date = check_content.find("h5").get_text().strip()

            # Check Title
            title = check_content.find("h2").get_text().strip()

            # Check Author
            author_str = check_content.find("h4").get_text()

            author = " ".join(author_str.split())

            author = author.replace("Por", " ").strip()

            # Check Subtitle
            subtitle = check_content.find("h3").get_text()

            # Check Entity
            div_entity = check_soup.find("div", {"class": "personaje-chequeo"})

            if div_entity != None:

                entity = div_entity.find("a").get_text()
            else:
                entity = "Pas d'entité"

            # Check Text
            div_text = check_content.find("div")

            text = ""
            for p in div_text.find_all("p"):
                text += p.get_text() + "\n"

            print("Test", title, author, date,
                  subtitle, check_value, entity)

            checks_content.append(
                [title, author, date, subtitle, check_value, entity, text])

        with open('colombiachecks.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=";")

            writer.writerow(["Title", "Author", "Date",
                             "Subtitle", "Value", "Entity", "Text"])
            writer.writerows(checks_content)

        # print(checks_links)


url_list = [
    'https://www.animalpolitico.com/sabueso/?seccion=discurso',
    'https://colombiacheck.com/chequeos',
]


for i in url_list:

    j = Extractor(i)
    print(j.key)
