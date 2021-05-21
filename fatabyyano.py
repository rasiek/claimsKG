# -*- coding: utf-8 -*-
import json
import math
import re
import sys
import dateparser
from typing import *

from bs4 import BeautifulSoup

from claim_extractor import Claim, Configuration
from claim_extractor.extractors import FactCheckingSiteExtractor, caching


class FatabyyanoFactCheckingSiteExtractor(FactCheckingSiteExtractor):
    # Constants

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)

    def get(self, url):
        """ @return the webpage """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        html = caching.get(url, headers=headers)
        soup = BeautifulSoup(html, 'lxml')
        # removing some useless tags
        for s in soup.select("script, iframe, head, header, footer, style"):
            s.extract()
        return soup

    def retrieve_listing_page_urls(self) -> List[str]:
        """
            Abstract method. Retrieve the URLs of pages that allow access to a paginated list of claim reviews. This
            concerns some sites where all the claims are not listed from a single point of access but first
            categorized by another criterion (e.g. on politifact there is a separate listing for each possible rating).
            :return: Return a list of listing page urls
        """
        
            
        different_urls = []
        different_rating_value = ["religious_related_rumors","medical-rumors","technological_related_rumors"
        ,"social_related_rumors","politics_related_rumors","fatabyyano_articles","science_related_rumors"]
        url_begin = "https://fatabyyano.net/category/"
        for value in different_rating_value:
            different_urls.append(url_begin + value + "/")
        return different_urls

    def find_page_count(self, parsed_listing_page: BeautifulSoup) -> int:
        """
            A listing page is paginated and will sometimes contain information pertaining to the maximum number of pages
            there are. For sites that do not have that information, please return a negative integer or None
            :param parsed_listing_page:
            :return: The page count if relevant, otherwise None or a negative integer
        """
        page_numbers = parsed_listing_page.select("div.nav-links a.page-numbers span")
        maximum = 1
        for page_number in page_numbers:
            p = int(page_number.text)
            if (p > maximum):
                maximum = p
        return 2

    def retrieve_urls(self, parse_page: BeautifulSoup) -> \
            List[str]:
        """
            :parsed_listing_page: --> une page (parsed) qui liste des claims
            :listing_page_url:    --> l'url associé à la page ci-dessus
            :number_of_page:      --> number_of_page
            :return:              --> la liste des url de toutes les claims
        """
        """
        result = []
        
        listing_category = ["religious_related_rumors","medical-rumors","technological_related_rumors","social_related_rumors","politics_related_rumors","fatabyyano_articles","science_related_rumors"]
        for category in listing_category:
            #print("fafa",category)
            #print(number_of_pages)
            url_begin = "https://fatabyyano.net/category/"+category+"/"
            #print(url_begin)
            
            result.append(url_begin)
        #print(result)
        return result
                #return json.loads(json_params)
        """

        #url_begin = listing_page_url + "page/"
        #url_end = "/"
        result = []
        
        listing_category = ["religious_related_rumors","medical-rumors","technological_related_rumors","social_related_rumors","politics_related_rumors","fatabyyano_articles","science_related_rumors"]
        
        for category in listing_category:
                        
            
            nbPage = 2
                
            while nbPage < 6: #while True:
                json_params = """{"columns":"3","exclude_items":"none","img_size":"default","ignore_items_size":false,"items_layout":"15632","items_offset":"1","load_animation":"none","overriding_link":"none","post_id":15837,"query_args":{"category_name":\""""+category+"""\","post_type":["post"],"post_status":["publish"],"tax_query":[{"taxonomy":"category","terms":[\""""+category+"""\"],"field":"slug","operator":"IN","include_children":true}],"paged":"""+str(nbPage)+"""},"orderby_query_args":{"orderby":{"date":"DESC"}},"type":"masonry","us_grid_ajax_index":1,"us_grid_filter_params":null,"us_grid_index":1,"_us_grid_post_type":"current_query"}"""
                data = json.loads(json_params)
    
    
                parsed = caching.post(f"https://fatabyyano.net/category/{category}", data=data,headers=self.headers)
                if parsed == None:
                    break 
                else:
                    parse_page= BeautifulSoup(parsed, "lxml")
                    articles_list = parse_page.findAll("article")
                    for article in articles_list:
                        url = article.find("a")["href"]
                        result.append(url)

         
                nbPage += 1
        
        return result
        

        """

        result = []
        url_end=""
        listing_category = ["religious_related_rumors","medical-rumors","technological_related_rumors","social_related_rumors","politics_related_rumors","fatabyyano_articles","science_related_rumors"]
        for category in listing_category:
            #print("fafa",category)
            #print(number_of_pages)
            url_begin = "https://fatabyyano.net/category/"+category+"/"
            #print(url_begin)
            for nbPage in range(number_of_pages):
                nbpages=str(nbPage)
                url_end = "{\"columns\":\"3\",\"exclude_items\":\"none\",\"img_size\":\"default\",\"ignore_items_size\":false,\"items_layout\":\"15632\",\"items_offset\":\"1\",\"load_animation\":\"none\",\"overriding_link\":\"none\",\"post_id\":15837,\"query_args\":{\"category_name\":\""+category+"\",\"post_type\":[\"post\"],\"post_status\":[\"publish\"],\"tax_query\":[{\"taxonomy\":\"category\",\"terms\":[\"category\"],\"field\":\"slug\",\"operator\":\"IN\",\"include_children\":true}],\"paged\":"+nbpages+"},\"orderby_query_args\":{\"orderby\":{\"date\":\"DESC\"}},\"type\":\"masonry\",\"us_grid_ajax_index\":1,\"us_grid_filter_params\":null,\"us_grid_index\":1,\"_us_grid_post_type\":\"current_query\"}"
                
                url = url_begin + url_end
                #print(url)
                #parsed_web_page = self.get(url)
            
                result.append(url)
        
        return result
         """       
    def extract_claim_and_review(self, parsed_claim_review_page: BeautifulSoup, url: str) -> List[Claim]:
        self.claim = self.extract_claim(parsed_claim_review_page)
        self.review = self.extract_review(parsed_claim_review_page)

        claim = Claim()
        claim.set_rating_value(
            self.extract_rating_value(parsed_claim_review_page))
        claim.set_rating(FatabyyanoFactCheckingSiteExtractor.translate_rating_value(
            self.extract_rating_value(parsed_claim_review_page)))
        claim.set_source("fatabyyano")
        claim.set_author("fatabyyano")
        claim.set_date_published(self.extract_date(parsed_claim_review_page))
        claim.set_claim(self.claim)
        claim.set_body(self.review)
        claim.set_refered_links(self.extract_links(parsed_claim_review_page))
        claim.set_title(self.extract_claim(parsed_claim_review_page))
        claim.set_date(self.extract_date(parsed_claim_review_page))
        claim.set_url(url)
        claim.set_tags(self.extract_tags(parsed_claim_review_page))

        return [claim]

    def is_claim(self, parsed_claim_review_page: BeautifulSoup) -> bool:
        return True

    def extract_claim(self, parsed_claim_review_page: BeautifulSoup) -> str:
        claim = parsed_claim_review_page.find('h1',{"class":"w-post-elm"})
        
        if claim:
           print("CLAIM: "+claim.text)
           return claim.text
            
        else:
            # print("something wrong in extracting claim")
            return "CLAIMNONTROUVER"

    def extract_review(self, parsed_claim_review_page: BeautifulSoup) -> str:
        #return self.escape(parsed_claim_review_page.select_one(
        liste = []
        article= parsed_claim_review_page.select("div.w-post-elm.post_content p")
        for a in article:
            
            revue = self.escape(a.text)
            liste.append(revue)
            
        print(liste[:])
        return liste
        

            
        

       


        

    def extract_date(self, parsed_claim_review_page: BeautifulSoup) -> str:
        
        d = parsed_claim_review_page.find("time",{"class":"w-post-elm"})
        
        if d:
            
            date = d["datetime"].split("T")[0]
            date_str = dateparser.parse(date).strftime("%Y-%m-%d")
            da = str(date_str)
            print(da)
            return da
            
        
        else:
            print("something wrong in extracting the date")
            return "dateNonTrouver"
        
    
     
          

    def extract_tags(self, parsed_claim_review_page: BeautifulSoup) -> str:
        """
            :parsed_claim_review_page:  --> the parsed web page of the claim
            :return:                    --> return a list of tags that are related to the claim
        """
        tags_link = parsed_claim_review_page.select(
            "div.w-post-elm.post_taxonomy.style_simple a[rel=\"tag\"]")
        tags = ""
        for tag_link in tags_link:
            if tag_link.text:
                tag = (tag_link.text).replace("#", "")
                tags += tag + ","

        return tags[:len(tags) - 1]

    def extract_author(self, parsed_claim_review_page: BeautifulSoup) -> str:
        return "fatabyyano"

    def extract_rating_value(self, parsed_claim_review_page: BeautifulSoup) -> str:
        numImage = 20000
        while numImage < 40000:
            e = parsed_claim_review_page.select_one("div.w-post-elm.post_content")
            value= e.select_one(f"img.alignnone.size-full.wp-image-{numImage}.lazyloaded")
            numImage += 1
        
            print(value)
            if value:
                
                v= value["alt"]
                print(v)
                
                return v
            else:
                # print("Something wrong in extracting rating value !")
                print("veraciteNonTROUVER")
                return "veraciteNonTrouver"
        

    def extract_links(self, parsed_claim_review_page: BeautifulSoup) -> str:
    
    
        links_tags = parsed_claim_review_page.find("article",{"class":"w-grid-item"})
        
        lien = links_tags.find("h2",{"class":"w-post-elm"})
        print(lien)
        if lien:
            liens = lien.find("a")["href"]
            print(liens)
            return liens       

        

    @staticmethod
    def translate_rating_value(initial_rating_value: str) -> str:
        return {
            "veraciteNonTrouver": "NULL",
            "صحيح": "TRUE",
            "زائف جزئياً": "MIXTURE",
            "عنوان مضلل": "OTHER",  # ?
            "رأي": "OTHER",  # ? (Opinion)
            "ساخر": "OTHER",  # ? (Sarcastique)
            "غير مؤهل": "FALSE",  # ? (Inéligible)
            "خادع": "FALSE",  # ? (Trompeur)
            "زائف": "FALSE" #faut
            
        }[initial_rating_value]

    # write this method (and tagme, translate) in an another file cause we can use it in other websites
    @staticmethod
    def get_json_format(tagme_entity):
        '''
            :tagme_entity: must be an object of AnnotateResposte Class returned by tagme function
        '''
        data_set = []
        i = 0
        min_rho = 0.1

        for annotation in tagme_entity.get_annotations(min_rho):
            entity = {}
            entity["id"] = annotation.entity_id
            entity["begin"] = annotation.begin
            entity["end"] = annotation.end
            entity["entity"] = annotation.entity_title
            entity["text"] = annotation.mention
            entity["score"] = annotation.score
            entity["categories"] = []
            if tagme_entity.original_json["annotations"][i]["rho"] > min_rho and "dbpedia_categories" in \
                    tagme_entity.original_json["annotations"][i]:
                for categorie in tagme_entity.original_json["annotations"][i]["dbpedia_categories"]:
                    entity["categories"].append(categorie)
            i = i + 1
            data_set.append(entity)

        return json.dumps(data_set)

    @staticmethod
    def cut_str(str_list):
        # cut string
        result_list = []

        for string in str_list:
            middle = math.floor(len(string) / 2)
            before = string.rindex(' ', 0, middle)
            after = string.index(' ', middle + 1)

            if middle - before < after - middle:
                middle = before
            else:
                middle = after

            result_list.append(string[:middle])
            result_list.append(string[middle + 1:])

        return result_list
    
    @staticmethod
    def escape(str):
        str = re.sub('[\n\t\r]', ' ', str)  # removing special char
        str = str.replace('"', '""')  # escaping '"' (CSV format)
        str = re.sub(' {2,}', ' ', str).strip()  # remoing extra spaces
        str = '"' + str + '"'
        return str

