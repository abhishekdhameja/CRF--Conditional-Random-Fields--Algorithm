# -*- coding: utf-8 -*-
import scrapy


class IndeedSpider(scrapy.Spider):
    name = 'indeed'
    max=7
    start_urls = ['https://www.indeed.com/jobs?q=software+engineer&l=San+Francisco%2C+CA&start=0']

    def parse(self, response):
		
		if 'start='in response.url:
			for jobUrl in response.xpath("//td[@id='resultsCol']//a[@data-tn-element='jobTitle']//@href").extract():
				if jobUrl is not None:
					yield response.follow("https://www.indeed.com"+jobUrl, callback= self.parse)
			nextURL=response.xpath("//div[@class='pagination']//a/@href").extract()[-1]
			if nextURL is not None:
				self.max-=1
				if self.max>0:
					yield response.follow("https://www.indeed.com"+nextURL, callback= self.parse)
		else:
			desc=""
			for str in response.xpath("//span[@id='job_summary']//text()").extract():
				desc=desc+" "+str
			yield {"job_desc": desc}