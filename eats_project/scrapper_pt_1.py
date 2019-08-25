# https://www.faceprep.in/tcs-ninja-exam-pattern-and-syllabus/
import pandas as pd
from selenium import webdriver


def rest_types_buttons():
	'''This function returns all the buttons for navigating each category'''
    
    buttons = wd.find_elements_by_xpath('//span[@class="zred"]')
    return buttons


def name_link_add():
	'''This function returns the link, name and address of restaurant for each page'''
    
    rest_link = []
    rest_name = []
    restaurants = wd.find_elements_by_xpath('//a[@class="result-title hover_feedback zred bold ln24   fontsize0 "]')
    for name in restaurants:
        rest_link.append(name.get_attribute('href'))
        rest_name.append(name.text)
    restaurants_address =  wd.find_elements_by_xpath('//div[@class="col-m-16 search-result-address grey-text nowrap ln22"]')
    rest_address = []
    for rest_add in restaurants_address:
        rest_address.append(rest_add.text)
    return rest_link, rest_name, rest_address

def get_data_rest_type(rest_type):
	'''this function returns the all the data from an individual category (all pages combined)'''
    
    try:
        prev_link,prev_name,prev_add = None, None, None
        rest_link, rest_name, rest_address = name_link_add()
        link = []
        name = []
        address = []
        while(prev_link != rest_link):
            prev_link,prev_name,prev_add = rest_link, rest_name, rest_address
            link = link + prev_link
            name = name + prev_name
            address = address + prev_add
            next_page_button = wd.find_element_by_xpath('//i[@class="right angle icon"]')
            next_page_button.click()
            wd.switch_to.window(wd.window_handles[0])
            rest_link, rest_name, rest_address = name_link_add()
            # Below two if conditions are for debugging
            if((len(rest_address) == len(rest_link) == len(rest_name)) == False):
                print("need to see, name link address mismatch")
                break
            if(len(rest_name) == 0):
                print("Empty found")
    except:
        print("unknown error")
    return link, name, address


def get_web_driver(url_link):
	
	url = "https://www.zomato.com/jamshedpur"
	url = str(url_link) 

	wd = webdriver.Firefox()
	wd.get(url)

	return wd

def get_areas(url_link):
	# Get all the areas in a city

	wd = get_web_driver(url_link)
	area_types = wd.find_element_by_xpath('//div[@class="ui segment row"]')
	area_types = area_types.text.split("\n")
	print(area_types)
	wd.quit()

	return area_types

def get_data(url_link):

	area_types = get_areas(url_link)
	
	for i in range(len(area_types)):

		city = str(area_types[i]).partition('(')[0]
		city = city[:-1]
		print(city)

		wd = webdriver.Firefox()
		wd.get(url)

		areaId = "Restaurants in " + city

		wd.find_element_by_xpath("//a[@class='col-l-1by3 col-s-8 pbot0'][contains(@title,'" + areaId + "')]").click()
		curr_url = wd.current_url
		print(curr_url)

		wd.get(curr_url)

		rest_types = wd.find_element_by_xpath('//div[@class="ui vertical pointing menu subzone_category_container"]')
		rest_types = rest_types.text.split("\n")
		print(rest_types)

		for j in range(len(rest_types)):
			# For each of the category
			type_ = ("_".join(rest_types[j].lower().split(' ')))
			print(type_)

			wd.switch_to.window(wd.window_handles[0])
			buttons = rest_types_buttons()

			buttons[j].click()
			wd.switch_to.window(wd.window_handles[0])

			try:
				# wd.implicitly_wait(30)
				wd.find_element_by_xpath('//a[@title="Click here to exclude restaurants from nearby locations"]').click()
				# wd.implicitly_wait(0)
			except:
				print('No nearby location link exist')

			# Collect the data
			data_rest_type = get_data_rest_type(type_)

			wd.get(curr_url)

			# Forma dataframe
			add = pd.DataFrame({'link' : data_rest_type[0], 'name' : data_rest_type[1], 'address' : data_rest_type[2]} )

			# save the file in csv format
			filename = city + "_" + type_ + "_.csv"
			add.to_csv(filename, index=False, columns = ['link', 'address', 'name'])

		wd.quit()

