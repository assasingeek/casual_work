# https://www.faceprep.in/tcs-ninja-exam-pattern-and-syllabus/
import pandas as pd
from selenium import webdriver
import glob

files = glob.glob(("Dataset/phase_1_data/*.csv"))
print((files))

for file in files:

	data = pd.read_csv(file)
	links = data["link"]
	res_names = data["name"]

	wd = webdriver.Firefox()

	# Extracting Features
	on_order = []
	phone_num = []
	res_time = []
	cost = []
	res_rating = []
	res_votes = []
	res_cuisines = []
	res_book = []

	cuisine_ind = 0
	for url in links:
		wd.get(url)

		# Find whether online ordering available or not
		try:
			wd.find_element_by_xpath('//a[@class="ui large green button order-now-action o2pad-fix o2_link"]')
			on_order.append(True)
		except Exception as ex:
			try:
				wd.find_element_by_xpath('//div[@class="ui large disabled button order-now-action o2_closed o2pad-fix tac"]')
				# on_order.pop()
				on_order.append(True)
			except Exception as ex:
				on_order.append(False)

		# Get the phone number
		tele_ele = wd.find_element_by_xpath('//span[@class="tel"]')
		phone_num.append(tele_ele.get_attribute('aria-label'))

		# Get restaurant open timing
		res_ele = wd.find_element_by_xpath('//div[@class="medium"]')
		res_time.append(str(res_ele.get_attribute('innerHTML')).split(';')[2])

		# Get cost for 2 people
		try:
			res_cost = wd.find_element_by_xpath('//div[@class="res-info-detail"]//span[@tabindex="0"]')
			cost.append(str(res_cost.get_attribute('aria-label')).split('(')[0])
		except Exception as ex:
			cost.append('Not Available')

		# Get restaurant rating and number of votes
		try:
			vote = wd.find_element_by_xpath('//span[@class="mt2 mb0 rating-votes-div rrw-votes grey-text fontsize5 ta-right"]')
			res_votes.append(vote.get_attribute('aria-label'))

			rating = wd.find_element_by_xpath('//div[@class="rating_hover_popup res-rating pos-relative clearfix mb5"]\
												//div[@tabindex="0"]')
			res_rating.append(rating.get_attribute('aria-label'))

		except Exception as ex:
			res_rating.append('New')
			res_votes.append('0 votes')

		res_cuisines.append([])

		# Get type of cuisines
		try:
			all_cuisines = wd.find_elements_by_xpath('//div[@class="res-info-cuisines clearfix"]//a[@class="zred"]')
			for cuisine in all_cuisines:
				res_cuisines[cuisine_ind].append(cuisine.text)
		except Exception as ex:
			res_cuisines[cuisine_ind].append('Not Available')

		cuisine_ind = cuisine_ind + 1

		# Find if booking table available or not
		try:
			book = wd.find_element_by_xpath('//a[@id="booktable"]')
			res_book.append("Available")
		except Exception as ex:
			res_book.append('Not Available')


	df = pd.DataFrame(data=on_order, columns=['Online Order'])
	df['URL'] = links
	df['Timings'] = res_time
	df['Cost'] = cost
	df['Rating'] = res_rating
	df['Votes'] = res_votes
	df['Phone Number'] = phone_num
	df['Table Booking'] = res_book
	df['Cuisines'] = res_cuisines
	df['Name'] = res_names
	print(df)
	new_file_name = file[:-4] + 'extracted' + '_.csv'
	df.to_csv(new_file_name)
	
