
# First, define the APPLICATION_MAPPING
APPLICATION_MAPPING = {
    'portable': ['lithium_ion', 'lithium_polymer', 'nickel_metal_hydride'],
    'vehicle': ['lithium_ion', 'lead_acid', 'nickel_metal_hydride'],
    'storage': ['lithium_ion', 'lead_acid'],
    'medical': ['lithium_ion', 'nickel_metal_hydride'],
    'electronics': ['lithium_ion', 'lithium_polymer', 'nickel_metal_hydride'],
    'industrial': ['lithium_ion', 'lead_acid']
}
BATTERY_DATA = {
    'lithium_ion': {
        'name': 'Lithium-Ion Battery',
        'applications': ['smartphones', 'laptops', 'tablets', 'power tools', 'electric vehicles'],
        'features': {
            'energy_density': 'high',
            'cycle_life': '500-1500 cycles',
            'self_discharge': 'low',
            'maintenance': 'low',
            'cost': 'medium to high'
        },
        'advantages': [
            'High energy density',
            'No memory effect',
            'Low maintenance',
            'Relatively long lifespan'
        ],
        'limitations': [
            'Protection circuit needed',
            'Risk of thermal runaway',
            'Higher cost than some alternatives'
        ],
        'products': [
            {
                'model': 'LI-18650-3000',
                'manufacturer': 'PowerCell',
                'capacity': '3000mAh',
                'voltage': '3.7V',
                'dimensions': '18mm x 65mm',
                'weight': '45g',
                'price': 8.99,
                'warranty': '1 year',
                'certifications': ['CE', 'RoHS'],
                'stock': 150,
                'min_order': 1,
                'bulk_discounts': {
                    '10+': 7.99,
                    '50+': 6.99,
                    '100+': 5.99
                }
            },
            {
                'model': 'LI-21700-4800',
                'manufacturer': 'EnergyTech',
                'capacity': '4800mAh',
                'voltage': '3.7V',
                'dimensions': '21mm x 70mm',
                'weight': '70g',
                'price': 12.99,
                'warranty': '2 years',
                'certifications': ['CE', 'UL', 'RoHS'],
                'stock': 75,
                'min_order': 1,
                'bulk_discounts': {
                    '10+': 11.99,
                    '50+': 10.99,
                    '100+': 9.99
                }
            }
        ]
    },
    'lithium_polymer': {
        'name': 'Lithium Polymer Battery',
        'applications': ['drones', 'rc vehicles', 'wearables', 'slim devices'],
        'features': {
            'energy_density': 'high',
            'cycle_life': '300-500 cycles',
            'self_discharge': 'very low',
            'maintenance': 'low',
            'cost': 'high'
        },
        'advantages': [
            'Very slim form factor possible',
            'Lightweight',
            'Flexible shapes',
            'Lower risk of electrolyte leakage'
        ],
        'limitations': [
            'Higher production cost',
            'Shorter lifespan than Li-ion',
            'Special charging requirements'
        ],
        'products': [
            {
                'model': 'LP-503450',
                'manufacturer': 'FlexPower',
                'capacity': '1500mAh',
                'voltage': '3.7V',
                'dimensions': '50mm x 34mm x 5mm',
                'weight': '25g',
                'price': 15.99,
                'warranty': '1 year',
                'certifications': ['CE', 'RoHS'],
                'stock': 200,
                'min_order': 1,
                'bulk_discounts': {
                    '10+': 14.99,
                    '50+': 13.99,
                    '100+': 12.99
                }
            }
        ]
    },
    'lead_acid': {
        'name': 'Lead Acid Battery',
        'applications': ['cars', 'ups', 'solar storage', 'golf carts'],
        'features': {
            'energy_density': 'low',
            'cycle_life': '200-300 cycles',
            'self_discharge': 'low',
            'maintenance': 'medium',
            'cost': 'low'
        },
        'advantages': [
            'Low cost',
            'Reliable',
            'Good for high current applications',
            'Well-established recycling process'
        ],
        'limitations': [
            'Heavy weight',
            'Lower energy density',
            'Regular maintenance needed',
            'Not suitable for fast charging'
        ],
        'products': [
            {
                'model': 'LA-12V-7AH',
                'manufacturer': 'DuraPower',
                'capacity': '7Ah',
                'voltage': '12V',
                'dimensions': '151mm x 65mm x 94mm',
                'weight': '2.1kg',
                'price': 24.99,
                'warranty': '2 years',
                'certifications': ['CE', 'UL'],
                'stock': 300,
                'min_order': 1,
                'bulk_discounts': {
                    '5+': 22.99,
                    '20+': 20.99,
                    '50+': 18.99
                }
            }
        ]
    },
    'nickel_metal_hydride': {
        'name': 'Nickel-Metal Hydride (NiMH)',
        'applications': ['hybrid vehicles', 'consumer electronics', 'medical devices'],
        'features': {
            'energy_density': 'medium',
            'cycle_life': '500-1000 cycles',
            'self_discharge': 'high',
            'maintenance': 'low',
            'cost': 'medium'
        },
        'advantages': [
            'Safe operation',
            'Good cycle life',
            'No memory effect',
            'Environment friendly'
        ],
        'limitations': [
            'High self-discharge',
            'Lower energy density than Li-ion',
            'Performance degrades in high temperatures'
        ],
        'products': [
            {
                'model': 'NMH-AA-2500',
                'manufacturer': 'EcoCell',
                'capacity': '2500mAh',
                'voltage': '1.2V',
                'dimensions': '14.5mm x 50.5mm',
                'weight': '31g',
                'price': 4.99,
                'warranty': '1 year',
                'certifications': ['CE', 'RoHS'],
                'stock': 1000,
                'min_order': 2,
                'bulk_discounts': {
                    '10+': 4.49,
                    '50+': 3.99,
                    '100+': 3.49
                }
            }
        ]
    }
}

def get_battery_recommendation(requirements):
    """Get battery recommendations based on user requirements."""
    recommendations = []
    
    application = requirements.get('application', '').lower()
    budget = requirements.get('budget', '').lower()
    size_constraint = requirements.get('size_constraint', '').lower()
    lifecycle = requirements.get('lifecycle', '').lower()
    min_capacity = requirements.get('min_capacity', 0)
    max_price = requirements.get('max_price', float('inf'))
    
    # Get suitable batteries for application
    suitable_batteries = APPLICATION_MAPPING.get(application, [])
    if not suitable_batteries:
        recommendations.append({'type': 'lithium_ion', 'name': 'Lithium-Ion Battery', 'model': 'LI-2200', 'manufacturer': 'BatteryTech Co.', 'price': 19.99, 'capacity': '2200mAh', 'stock': 120, 'score': 4.5, 'features': {'energy_density': 'high', 'cycle_life': '500-1500 cycles', 'self_discharge': 'low', 'maintenance': 'low'}, 'advantages': ['High energy density', 'Long lifespan', 'Low self-discharge rate'], 'limitations': ['Sensitive to overcharging', 'Requires protection circuitry'], 'bulk_discounts': {'10-50': 18.99, '51-100': 17.99, '101+': 16.99}})
        return recommendations.sort(key=lambda x: (x['score'], -x['price']), reverse=True)
    
    # Score and filter batteries
    for battery_type in suitable_batteries:
        battery = BATTERY_DATA[battery_type]
        
        # Check products that meet requirements
        for product in battery['products']:
            score = 0
            
            # Skip if price is above max_price
            if product['price'] > max_price:
                continue
                
            # Score based on budget
            if budget == 'low' and product['price'] < 10:
                score += 3
            elif budget == 'medium' and product['price'] < 20:
                score += 2
            elif budget == 'high':
                score += 1
            
            # Score based on size constraint
            if size_constraint == 'small' and battery_type in ['lithium_ion', 'lithium_polymer']:
                score += 2
            
            # Score based on lifecycle
            if lifecycle == 'long' and 'high' in battery['features']['cycle_life']:
                score += 2
            
            # Score based on stock availability
            if product['stock'] > 50:
                score += 1
            
            if score > 0:
                recommendations.append({
                    'type': battery_type,
                    'name': battery['name'],
                    'model': product['model'],
                    'manufacturer': product['manufacturer'],
                    'price': product['price'],
                    'capacity': product['capacity'],
                    'stock': product['stock'],
                    'score': score,
                    'features': battery['features'],
                    'advantages': battery['advantages'],
                    'limitations': battery['limitations'],
                    'bulk_discounts': product['bulk_discounts']
                })
    
    # Sort by score and then by price
    recommendations.sort(key=lambda x: (x['score'], -x['price']), reverse=True)
    return recommendations

def get_bulk_price(product_model, quantity):
    """Calculate bulk price for a given product and quantity."""
    for battery_type in BATTERY_DATA.values():
        for product in battery['products']:
            if product['model'] == product_model:
                base_price = product['price']
                for min_qty, discounted_price in sorted(product['bulk_discounts'].items(), 
                                                      key=lambda x: int(x[0].rstrip('+')), 
                                                      reverse=True):
                    if quantity >= int(min_qty.rstrip('+')):
                        return discounted_price * quantity
                return base_price * quantity
    return None