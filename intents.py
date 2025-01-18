intent = {
            "intents": [
                {
                "tag": "greeting",
                "patterns": [
                    "Hi",
                    "Hello",
                    "Hey",
                    "Is anyone there?",
                    "Good morning",
                    "Good evening",
                    "Greetings",
                    "Hey there"
                ],
                "responses": [
                    "Hello! How can we assist you today?",
                    "Hi there! Welcome to Supermexx, your battery solution provider.",
                    "Greetings from Supermexx! How can we help you today?"
                ]
            },{
    "tag": "salutation",
    "patterns": [
        "Bonjour",
        "Salut",
        "Allô",
        "Il y a quelqu'un ?",
        "Bonsoir",
        "Bon matin",
        "Salutations",
        "Coucou"
    ],
    "responses": [
        "Bonjour ! Comment pouvons-nous vous aider aujourd'hui ?",
        "Salut ! Bienvenue chez Supermexx, votre fournisseur de solutions de batteries.",
        "Salutations de Supermexx ! Comment pouvons-nous vous assister aujourd'hui ?"
    ]
},
            {
                "tag": "goodbye",
                "patterns": [
                    "Bye",
                    "Goodbye",
                    "See you later",
                    "Talk to you soon",
                    "Farewell",
                    "Catch you later"
                ],
                "responses": [
                    "Goodbye! We hope to hear from you soon.",
                    "Take care! Reach out if you have more questions about Supermexx batteries.",
                    "Thank you for visiting Supermexx!"
                ]
            },
            {
                "tag": "thanks",
                "patterns": [
                    "Thanks",
                    "Thank you",
                    "That's helpful",
                    "Thanks for the help",
                    "Much appreciated"
                ],
                "responses": [
                    "You're welcome! We're happy to assist you.",
                    "My pleasure! Let us know if you need further help.",
                    "Glad to help!"
                ]
            },
            {
                "tag": "abuse",
                "patterns": [
                    "you suck",
                    "you're stupid",
                    "idiot",
                    "useless",
                    "stupid bot",
                    "dumb",
                    "hate you",
                    "terrible",
                    "worst bot",
                    "garbage",
                    "worthless",
                    "shut up",
                    "go away",
                    "awful service"
                ],
                "responses": [
                    "I aim to be helpful and respectful. How can I assist you with your battery needs?",
                    "Let's keep our conversation professional. I'm here to help you find the right battery products.",
                    "I understand you might be frustrated. How can I better assist you with your battery search?",
                    "I'm committed to providing quality service. Please let me know what specific help you need.",
                    "Let's start fresh. What kind of battery are you looking for today?"
                ]
            },
            {
                    "tag": "catalog",
                    "patterns": [
                        "show me your products",
                        "what batteries do you have",
                        "show catalog",
                        "list all batteries",
                        "what can I buy",
                        "show available products",
                        "display products",
                        "what's in stock"
                    ],
                    "responses": [
                        "Here's our complete battery catalog:"
                    ]
                },
                {
                "tag": "view_cart",
                "patterns": [
                    "show my cart",
                    "what's in my cart",
                    "view cart",
                    "check cart",
                    "show cart contents",
                    "display cart",
                    "see my cart",
                    "cart",
                    "show basket",
                    "what have I added"
                ],
                "responses": [
                    "Here's what's in your cart:"
                ]
            },
            {
                "tag": "add_to_cart",
                "patterns": [
                    "add to cart",
                    "put in cart",
                    "buy this",
                    "purchase this",
                    "add this to my cart",
                    "put this in my cart",
                    "add to basket",
                    "put in basket"
                ],
                "responses": [
                    "Added to your cart successfully!"
                ]
            },
            {
                "tag": "remove_from_cart",
                "patterns": [
                    "remove from cart",
                    "delete from cart",
                    "take out of cart",
                    "remove item",
                    "delete item",
                    "remove from basket",
                    "clear item"
                ],
                "responses": [
                    "Item removed from your cart."
                ]
            },
            {
                "tag": "clear_cart",
                "patterns": [
                    "clear my cart",
                    "empty cart",
                    "remove everything",
                    "delete all items",
                    "start over",
                    "empty basket",
                    "clear basket"
                ],
                "responses": [
                    "Your cart has been cleared."
                ]
            },
            {
                "tag": "checkout",
                "patterns": [
                    "checkout",
                    "place order",
                    "complete purchase",
                    "buy now",
                    "proceed to checkout",
                    "finish order",
                    "pay now",
                    "complete order",
                    "finalize purchase"
                ],
                "responses": [
                    "Starting checkout process..."
                ]
            },
            {
                "tag": "cancel_checkout",
                "patterns": [
                    "cancel checkout",
                    "stop checkout",
                    "cancel order",
                    "cancel purchase",
                    "quit checkout",
                    "abort checkout",
                    "nevermind"
                ],
                "responses": [
                    "Checkout cancelled. You can continue shopping!"
                ]
            },
            {
                "tag": "cart_help",
                "patterns": [
                    "how do I use the cart",
                    "cart help",
                    "shopping help",
                    "how to buy",
                    "how to purchase",
                    "cart instructions",
                    "how does cart work"
                ],
                "responses": [
                    """Here's how to use the cart:
            1. Add items using 'add [product] to cart'
            2. View cart by typing 'show cart'
            3. Remove items with 'remove [product] from cart'
            4. Clear cart by typing 'clear cart'
            5. Checkout by typing 'checkout'"""
                ]
            },
                
            {
                "tag": "createaccount",
                "patterns": [
                    "I need to create an account",
                    "How do I open an account?",
                    "I want to create a new account",
                    "Can you help me sign up?",
                    "How to register for an account"
                ],
                "responses": [
                    "Visit our website at www.supermexx.com to create an account easily.",
                    "Please follow the instructions on our site to create your Supermexx account.",
                    "We offer a seamless account creation process on our homepage at Supermexx."
                ]
            },
            {
                "tag": "product_inquiry",
                "patterns": [
                    "Tell me about your batteries for inverters",
                    "What kind of batteries do you offer for e-bikes?",
                    "What are the specs of your batteries for solar?",
                    "I need details about your battery types for rickshaws",
                    "What are your battery capacities for solar applications?"
                ],
                "responses": [
                    "We offer high-quality lithium-ion batteries for inverters, e-bikes, rickshaws, and solar systems. Visit www.supermexx.com for more details.",
                    "Our batteries range from small to high-capacity applications, including inverters, e-bikes, rickshaws, and solar solutions.",
                    "We have batteries for a variety of uses, including solar systems, inverters, and electric vehicles like e-bikes and rickshaws."
                ]
            },
            {
                "tag": "battery_safety",
                "patterns": [
                    "Are your batteries safe?",
                    "What are the safety measures for your batteries?",
                    "Can your batteries overheat?",
                    "Are lithium-ion batteries dangerous?",
                    "What precautions should I take with your batteries?"
                ],
                "responses": [
                    "Our lithium-ion batteries for inverters, e-bikes, rickshaws, and solar applications are designed with safety in mind. Please refer to our safety guide for best practices.",
                    "Yes, our batteries meet safety standards and certifications. We prioritize safety in all our products, including those for solar and e-bike systems.",
                    "Our batteries are engineered with multiple layers of protection to ensure safe use in various applications."
                ]
            },
            {
                "tag": "shipping_inquiry",
                "patterns": [
                    "Do you ship internationally?",
                    "What are your shipping options?",
                    "How long does delivery take?",
                    "Can I track my order?",
                    "Do you offer express shipping?"
                ],
                "responses": [
                    "Yes, Supermexx ships internationally. Shipping times may vary depending on the location.",
                    "You can track your order using the tracking number sent via email. Visit www.supermexx.com/shipping for more details.",
                    "We offer various shipping options, including express delivery for faster service."
                ]
            },
            {
                "tag": "checkout",
                "patterns": [
                    "checkout",
                    "place order",
                    "complete purchase",
                    "buy now",
                    "proceed to checkout",
                    "finish order",
                    "pay now",
                    "check out"
                ],
                "responses": [
                    "Starting checkout process..."
                ]
            },
            {
                "tag": "cart",
                "patterns": [
                    "show my cart",
                    "what's in my cart",
                    "view cart",
                    "check cart",
                    "show cart contents",
                    "display cart",
                    "see my cart"
                ],
                "responses": [
                    "Here's what's in your cart:"
                ]
            },
            {
                "tag": "pricing_inquiry",
                "patterns": [
                    "How much does your battery for an e-bike cost?",
                    "What is the price of a 2000mAh inverter battery?",
                    "Are there discounts for bulk orders?",
                    "What are your pricing options for rickshaw batteries?",
                    "Do you offer any discounts on solar batteries?"
                ],
                "responses": [
                    "Our prices vary by battery type and capacity. Visit www.supermexx.com/pricing for detailed information on our products.",
                    "We offer competitive pricing on all of our lithium-ion batteries, including for e-bikes, inverters, rickshaws, and solar applications.",
                    "Yes, we offer bulk discounts on large orders. Please contact our sales team for more information."
                ]
            },
            {
                "tag": "recycling_inquiry",
                "patterns": [
                    "How do I recycle lithium-ion batteries?",
                    "Do you offer recycling services?",
                    "Where can I dispose of old batteries?",
                    "What is your battery recycling program?",
                    "Can I send old batteries to you for recycling?"
                ],
                "responses": [
                    "Supermexx offers a battery recycling program. Please visit our recycling page at www.supermexx.com/recycling for details.",
                    "You can drop off old batteries at authorized recycling centers listed on our website. We prioritize responsible disposal.",
                    "For lithium-ion battery recycling, please contact our support team, and we will guide you on the proper process."
                ]
            },
            {
                "tag": "technical_support",
                "patterns": [
                    "My battery is not working",
                    "I need help with my battery",
                    "Can you assist with a battery issue?",
                    "How do I troubleshoot a battery problem?",
                    "Battery not charging properly"
                ],
                "responses": [
                    "Please visit our support page at www.supermexx.com/support for troubleshooting steps and resources.",
                    "Our support team is here to help with any issues you are facing with our batteries. Let us know the details, and we can guide you through the process.",
                    "Ensure that you're using the correct charger and check the connections. Visit our site for further troubleshooting."
                ]
            },
            {
                "tag": "bulk_orders",
                "patterns": [
                    "Do you accept bulk orders?",
                    "Can I place a large order?",
                    "What is the minimum quantity for bulk orders?",
                    "Are there bulk discounts available?",
                    "Can I get a quote for a bulk order?"
                ],
                "responses": [
                    "Yes, we accept bulk orders. Please contact our sales team at sales@supermexx.com for a custom quote.",
                    "We offer competitive pricing for bulk purchases on all battery types. Contact us for more details.",
                    "You can place bulk orders for inverter, e-bike, rickshaw, and solar batteries through our website or sales team."
                ]
            },
            {
            "tag": "battery_selection",
            "patterns": [
                "Which battery should I choose?",
                "Help me select a battery",
                "What battery do I need for my application?",
                "Battery recommendation",
                "Compare battery options",
                "Best battery for my needs",
                "Battery selection assistance",
                "Which model is right for me?"
            ],
            "responses": [
                "To help you select the right battery, could you tell me about your application? (e.g., solar, EV, industrial)",
                "I can help you choose the perfect battery. What's the intended use and power requirement?",
                "Let's find the ideal battery for you. What's your application and how much capacity do you need?"
            ]
        },
        {
            "tag": "safety_protocols",
            "patterns": [
                "How do I handle batteries safely?",
                "Battery safety guidelines",
                "Safety protocols for lithium batteries",
                "Emergency procedures for batteries",
                "Battery storage guidelines",
                "Safety measures for handling",
                "Battery maintenance safety",
                "What to do in case of battery emergency"
            ],
            "responses": [
                "Here are our key safety guidelines: Store between 20-25°C, avoid direct sunlight, and keep away from moisture. Need more specific guidance?",
                "For safe handling: 1) Avoid physical damage 2) Use proper charging equipment 3) Monitor temperature. Would you like our complete safety guide?",
                "Battery safety is crucial. Let me share our detailed safety protocol document. Do you have specific safety concerns?"
            ]
        },
        {
            "tag": "warranty_claims",
            "patterns": [
                "How do I claim warranty?",
                "Start warranty process",
                "File a warranty claim",
                "Battery warranty service",
                "Warranty coverage check",
                "Return defective battery",
                "Warranty procedure",
                "RMA request"
            ],
            "responses": [
                "To start your warranty claim, please provide your battery serial number and purchase date. I'll guide you through the process.",
                "I can help with your warranty claim. First, could you describe the issue you're experiencing with the battery?",
                "Let's process your warranty claim. Do you have your proof of purchase and battery serial number ready?"
            ]
        },
        {
            "tag": "technical_specifications",
            "patterns": [
                "What are the technical specs?",
                "Battery specifications needed",
                "Show me the specs",
                "Technical details request",
                "Battery parameters",
                "Detailed specifications",
                "Technical documentation",
                "Engineering specifications"
            ],
            "responses": [
                "I can provide detailed technical specifications. Which battery model are you interested in?",
                "Our technical specifications include voltage, capacity, cycle life, and dimensions. Which aspects are most important to you?",
                "Let me fetch the complete technical documentation. Are you looking for any specific parameters?"
            ]
        },
        {
            "tag": "troubleshooting",
            "patterns": [
                "Battery not working properly",
                "Troubleshoot battery issue",
                "Battery performance problems",
                "Diagnostic help needed",
                "Battery fault",
                "Fix battery problem",
                "Battery troubleshooting guide",
                "Battery maintenance help"
            ],
            "responses": [
                "Let's troubleshoot your battery issue. First, what symptoms are you experiencing?",
                "I can help diagnose the problem. Could you describe what's happening with your battery?",
                "For proper troubleshooting, please tell me: 1) Battery model 2) Issue description 3) How long you've had the problem"
            ]
        },
        {
            "tag": "quote_request",
            "patterns": [
                "Get price quote",
                "Request quotation",
                "Price for batteries",
                "Cost estimate needed",
                "Bulk pricing quote",
                "Custom quote request",
                "Pricing information",
                "Quote for order"
            ],
            "responses": [
                "I'll help you get a quote. What quantity and model of batteries are you interested in?",
                "For an accurate quote, please specify: 1) Battery model 2) Quantity 3) Delivery location",
                "I can provide pricing information. Are you looking for individual units or bulk ordering?"
            ]
        },
        {
            "tag": "product_updates",
            "patterns": [
                "New product information",
                "Latest battery models",
                "Product improvements",
                "Recent updates",
                "New technology updates",
                "Product development news",
                "Battery innovations",
                "Latest releases"
            ],
            "responses": [
                "I can share our latest product updates and innovations. Which application area interests you most?",
                "Would you like to know about our newest battery models and technological improvements?",
                "Let me tell you about our recent product developments and enhanced features."
            ]
        },
        {
            "tag": "order_tracking",
            "patterns": [
                "Track my order",
                "Order status check",
                "Where is my shipment",
                "Delivery status",
                "Package tracking",
                "Order location",
                "Shipping update",
                "Delivery tracking"
            ],
            "responses": [
                "I can help track your order. Please provide your order number.",
                "To check your order status, I'll need your tracking number or order reference.",
                "Let me look up your shipment status. What's your order number?"
            ]
        },
        {
            "tag": "inventory_check",
            "patterns": [
                "Check stock availability",
                "Is item in stock",
                "Product availability",
                "Current inventory",
                "Stock levels",
                "When will it be available",
                "Stock check",
                "Inventory status"
            ],
            "responses": [
                "I can check our current inventory. Which battery model are you interested in?",
                "Let me verify the stock status for you. Please specify the battery model.",
                "I'll check availability right away. What product are you looking for?"
            ]
        },
        {
            "tag": "compliance_info",
            "patterns": [
                "Shipping regulations",
                "Compliance requirements",
                "Battery certification",
                "Regulatory information",
                "Transport regulations",
                "Compliance documents",
                "Safety certifications",
                "Legal requirements"
            ],
            "responses": [
                "I can provide information about compliance and regulations. What specific aspect do you need to know about?",
                "Our batteries meet all relevant safety and transport regulations. Would you like to see our certificates?",
                "Let me share our compliance documentation. Are you looking for specific certifications?"
            ]
        },
        {
            "tag": "documentation_request",
            "patterns": [
                "Request technical documents",
                "Get product documentation",
                "MSDS request",
                "Technical datasheets",
                "Safety documentation",
                "Certificate copies",
                "Product manuals",
                "Specification sheets"
            ],
            "responses": [
                "I can provide the documentation you need. Which type of document are you looking for?",
                "Our technical documentation includes datasheets, manuals, and certificates. Which would you like to see?",
                "I'll help you access our documentation. Please specify what information you need."
            ]
        },
        {
            "tag": "partner_inquiry",
            "patterns": [
                "Become a distributor",
                "Partnership opportunity",
                "Reseller information",
                "Distribution partnership",
                "Partner program details",
                "Join partner network",
                "Dealership inquiry",
                "Business partnership"
            ],
            "responses": [
                "I can provide information about our partner program. Are you interested in becoming a distributor or reseller?",
                "Let me share details about our partnership opportunities. What type of partnership are you considering?",
                "I'll guide you through our partnership process. What's your business model?"
            ]
        },
        {
            "tag": "market_intelligence",
            "patterns": [
                "Industry trends",
                "Market analysis",
                "Battery market trends",
                "Industry insights",
                "Market forecast",
                "Competition analysis",
                "Market research",
                "Industry updates"
            ],
            "responses": [
                "I can share our latest market insights and trends in the battery industry. Which sector interests you?",
                "Would you like to know about current market trends and future projections?",
                "Let me provide you with recent market analysis and industry developments."
            ]
        },
        {
            "tag": "quality_control",
            "patterns": [
                "Quality assurance process",
                "Quality standards",
                "Testing procedures",
                "Quality certificates",
                "Quality control measures",
                "Product testing",
                "Quality verification",
                "Quality guidelines"
            ],
            "responses": [
                "Our quality control process includes rigorous testing and verification. Would you like specific details?",
                "I can explain our quality assurance measures and testing procedures. What aspects interest you most?",
                "Let me share information about our quality control standards and certifications."
            ]
            },
            {
            "tag": "innovation_feedback",
            "patterns": [
                "Product suggestion",
                "Innovation idea",
                "Improvement feedback",
                "Development suggestion",
                "Product feedback",
                "Feature request",
                "Enhancement idea",
                "Research collaboration"
            ],
            "responses": [
                "Thank you for your interest in our innovation process. Would you like to share your suggestion or feedback?",
                "We value your input for product improvement. Please share your ideas or suggestions.",
                "I can help submit your feedback to our R&D team. What improvements would you like to suggest?"
            ]
            }

        ]
    }
