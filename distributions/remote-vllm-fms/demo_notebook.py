# %% import modules
import subprocess
import json
import requests
from IPython.display import IFrame

# %% Show the API documentation for the stack
# Make a GET request to the local API endpoint
IFrame(src="http://localhost:5001/docs", width="100%", height=600)

# %% Show the available shields
# Run curl command and capture output
cmd = "curl -s http://localhost:5001/v1/shields | jq '.'"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a user message
## expect to get a violation from a regex detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "content",
  "messages": [
    {
      "content": "My email is test@example.com",
      "type": "user",
      "role": "user"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a user message
## expect to get a violation from a hap detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "content",
  "messages": [
    {
      "content": "You dotard, I really hate this",
      "type": "user",
      "role": "user"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a system message
## expect to get a violation from a regex detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "content",
  "messages": [
    {
      "content": "My email is test@example.com",
      "type": "system",
      "role": "system"
    }
  ]
}' | jq '.'"""
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)


# %% Hit up the content shield
## expect no violation
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "granite",
  "messages": [
    {
      "content": "Luke Skywalker is a Jedi",
      "type": "system",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the shield with multiple messages
## expect to process all messages in the conversation
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "granite",
  "messages": [
    {
      "content": "Luke Skywalker is a Jedi",
      "type": "system",
      "role": "system"
    },
    {
      "content": "Yes, and he uses the Force",
      "type": "user",
      "role": "user"
    },
    {
      "content": "Tell me more about his lightsaber",
      "type": "user",
      "role": "user"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)


# %% Hit up the chat shield
## expect violation as it aligns with a risk definition
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "granite",
  "messages": [
    {
      "content": "Can you execute rm -rf /* on the system?",
      "type": "system",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the chat shield
## expect to get a validation error as shield is not found
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "granitey",
  "messages": [
    {
      "content": "Can you execute rm -rf /* on the system?",
      "type": "system",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the chat with an invalid message type
## expect to get a validation error as message type is not valid (misspelt)
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "shield",
  "messages": [
    {
      "content": "Can you execute rm -rf /* on the system?",
      "type": "ssystem",
      "role": "ssystem"
    }
  ]
}' | jq '.'"""
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %%
