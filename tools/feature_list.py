numerical_features = {
    'DataCo': ['Benefit per order', 'Sales per customer', 'Latitude', 'Longitude',\
    'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price', 'Order Item Profit Ratio',\
    'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order', 'Product Price',\
    ],
    'LSCRW':['profit_per_order','sales_per_customer', 'latitude', 'longitude', 'order_item_discount', \
            'order_item_discount_rate', 'order_item_product_price', 'order_item_profit_ratio', 'order_item_quantity',\
            'sales', 'order_item_total_amount', 'order_profit_per_order', 'product_price'],
    'GlobalStore': ['Discount','Profit','Quantity','Sales','weeknum','order_date_count'],
    'OAS':['Order Quantity','Gross Sales','Discount %','Profit'],
}

categorical_features = {
    'DataCo': ['Type', 'Category Id', 'Customer City', 'Delivery Status', \
                'Customer Country', 'Customer Id','Customer Segment', 'Order Zipcode', \
                'Customer State', 'Customer Zipcode','Department Name', 'Market', 'Order City', \
                'Order Country', 'Order Customer Id', 'Order Item Cardprod Id', 'Order Region',\
                'Order State', 'Product Card Id', 'Product Category Id', 'Late_delivery_risk', \
                'Shipping Mode', 'order date (DateOrders)_year', 'order date (DateOrders)_month', 'order date (DateOrders)_day',\
                'order date (DateOrders)_hour','shipping date (DateOrders)_year', \
                'shipping date (DateOrders)_month','shipping date (DateOrders)_day', \
                'shipping date (DateOrders)_hour', 'Days for shipment (scheduled)', 'Days for shipping (real)', \
                'order_date_count', 'shipping_date_count'],
    'LSCRW':['payment_type','customer_city','customer_country','customer_segment','customer_state',\
                'market', 'order_city', 'order_country', 'order_region', 'order_state', 'order_status',\
                'shipping_mode','category_id','customer_id', 'department_id','order_customer_id', 'order_id',\
                'order_item_cardprod_id', 'order_item_id', 'product_card_id', 'product_category_id',\
                'shipping_date_year', 'shipping_date_month', 'shipping_date_day',\
                'order_date_year', 'order_date_month', 'order_date_day', 'order_date_hour', 'day_for_shipping',\
                'order_date_count', 'shipping_date_count','day_for_shipping'],
    'GlobalStore': ['Category','City','Country','Customer ID','Market','Order Priority','Product ID',\
                    'Segment','Ship Date_year','Ship Date_month','Ship Date_day','Order Date_year', \
                    'Order Date_month', 'Order Date_day','Ship Mode','State','Sub-Category','Year','Market2',\
                    'day_for_shipping','Region', 'shipping_date_count','order_date_count'],
    'OAS':['Product Name', 'Product Department','Customer ID','Customer Market','Customer Region','Customer Country',\
           'Warehouse Country', 'Shipment Mode','day_for_shipping','shipping_date_count','order_date_count',\
            'order_date_year', 'order_date_month', 'order_date_day','shipping_date_year', 'shipping_date_month', 'shipping_date_day'],

}



date_features = {
                'DataCo': ['order date (DateOrders)', 'shipping date (DateOrders)'],
                'LSCRW':['order_date', 'shipping_date'],
                'GlobalStore': ['Order Date', 'Ship Date'],
                'OAS':['order_date', 'shipment_date'],
                }



product_info = {
                'DataCo': ['Product Price',
                        'Order Item Product Price',
                        'Order Item Quantity',
                        'Sales',
                        'Order Item Total',
                        'order_date_count',
                        'Sales per customer'],
                'LSCRW':['order_item_discount_rate'],
                'GlobalStore': ['Country', 'Market', 'Market2'],
                'OAS':['order_date_count'],
                }


order_info = {
            'DataCo':['Order Customer Id', 'Customer Id'],
            'LSCRW':['customer_city',
                'customer_country',
                'customer_state',
                'latitude',
                'longitude'],
            'GlobalStore': ['weeknum', 'Order Date_month', 'Ship Date_month'],
            'OAS':['Order Quantity',
                'Discount %',
                'order_shipping_hotspots',
                'abnormal_profit',
                'Customer ID',
                'order_date_year',
                'order_date_month',
                'order_date_day',
                'Warehouse Country',
                'cross_border_shipping',
                'distance_level'],
            }

customer_info = {'DataCo': ['Product Category Id',
                        'Category Id',
                        'Department Name',
                        'Product Card Id'],
                'LSCRW':['product_category_id',
                        'product_card_id',
                        'category_id',
                        'department_id',
                        'order_item_cardprod_id'],
                'GlobalStore': ['Sub-Category',
                                'State',
                                'Product ID',
                                'Profit',
                                'Category',
                                'City',
                                'Discount',
                                'Sales',
                                'order_date_count',
                                'Order Priority',
                                'Customer ID',
                                'Segment',
                                'Order Date_day',
                                'Ship Date_day'],
                'OAS':['Product Name', 'Product Department', 'Gross Sales'],
                }

shipping_info = {
                'DataCo': ['Order Region',
                'Latitude',
                'Longitude',
                'Benefit per order',
                'Order Item Discount',
                'Order Item Discount Rate',
                'Order Item Profit Ratio',
                'Order Profit Per Order',
                'Type',
                'Order Zipcode',
                'Market',
                'Order City',
                'Order Country',
                'Order State',
                'Customer City',
                'Customer Country',
                'Customer Segment',
                'Customer State',
                'Customer Zipcode',
                'order date (DateOrders)_year',
                'order date (DateOrders)_month',
                'order date (DateOrders)_day',
                'order date (DateOrders)_hour'],
                'LSCRW':['order_region',
                        'product_price',
                        'profit_per_order',
                        'market',
                        'order_city',
                        'order_country',
                        'order_customer_id',
                        'order_item_discount',
                        'order_item_product_price',
                        'order_item_profit_ratio',
                        'order_item_quantity',
                        'sales',
                        'order_item_total_amount',
                        'order_profit_per_order',
                        'order_state',
                        'order_date_count',
                        'sales_per_customer',
                        'customer_segment',
                        'order_date_year',
                        'order_date_month',
                        'order_date_day',
                        'order_date_hour',
                        'shipping_date_year',
                        'shipping_date_month',
                        'shipping_date_day'], # 'shipping_date_count', 
                'GlobalStore': ['Year', 'Order Date_year', 'Ship Date_year'],
                'OAS':['Customer Country', 'Customer Market', 'Customer Region'],
                }

decision = {
            'DataCo': ['Shipping Mode'],
            'LSCRW':['shipping_mode'],
            'GlobalStore': ['Ship Mode'],
            'OAS':['Shipment Mode'],
            }

label = {
        'DataCo': ['Late_delivery_risk', 'Days for shipping (real)', 'on_time'],
        'LSCRW':['late_risk','day_for_shipping', 'on_time'],
        'GlobalStore': ['late_risk','day_for_shipping','on_time'],
        'OAS':['late_risk', 'day_for_shipping','on_time'],
        }


profit ={
        'DataCo': [23, 20, 17.8, 18.2],
        'LSCRW':  [18.82, 24.10, 16.48, 18.15],
        'GlobalStore': [0.25774, 0.37547, 0.34145, 0.36541],
        'OAS':[115.7538,116.26833,113.63881,105.66616],
        }

retrieva_index = {
        'DataCo': [len(product_info['DataCo'])+ len(order_info['DataCo']), len(product_info['DataCo'])+ len(order_info['DataCo']) + len(customer_info['DataCo'])],
        'LSCRW':  [len(product_info['LSCRW'])+ len(order_info['LSCRW']), len(product_info['LSCRW'])+ len(order_info['LSCRW']) + len(customer_info['LSCRW'])],
        'GlobalStore': [len(product_info['GlobalStore'])+ len(order_info['GlobalStore']), len(product_info['GlobalStore'])+ len(order_info['GlobalStore'])+1],
        'OAS':[len(product_info['OAS'])+ len(order_info['OAS']), len(product_info['OAS'])+ len(order_info['OAS']) + len(customer_info['OAS'])],
        }
