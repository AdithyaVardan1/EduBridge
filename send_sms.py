from twilio.rest import Client

# Your Twilio account SID and auth token
account_sid = 'ACe2c36bbf07c58f432307c3f455fe8baf'
auth_token = '2e06d9841d6156c1f2ba86c31aca158e'

# Initialize the Twilio client
client = Client(account_sid, auth_token)

# Sending the SMS
message = client.messages.create(
    to="+917410003555",  # Replace with the recipient's phone number
    from_="+15183133614",  # Replace with your Twilio phone number
    body="Hello, this is a test message from Adithya Gudise!")

print(message.sid)
