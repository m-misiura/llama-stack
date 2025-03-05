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

# %% Hit up the content shield with a system message
## expect to get a violation from a regex detector; no violation from a hap detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "My email is test@example.com",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a system message
## expect to get a violation from a hap detector; no violation from a regex detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "You dotard, I really hate this",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a system message
## expect to get a violation from both a regex detector and a hap detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "You dotard, I really hate this and my email is test@ibm.com",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a system message
## expect no violation from neither a regex detector nor a hap detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "This is a test message",
      "role": "user"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a list of system message
## expect a mixture of violations and no violations
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "You dotard, I really hate this", 
      "role": "system"
    },
    {
      "content": "My email is test@ibm.com",
      "role": "system"
    },
    {
      "content": "This is a test message",
      "role": "system"
    }
  ]
}' | jq '.'"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a list of system and user messages
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "You dotard, I really hate this", 
      "role": "system"
    },
    {
      "content": "My email is test@ibm.com",
      "role": "system"
    },
    {
      "content": "This is a test message",
      "role": "user"
    }
  ]
}' | jq '.'
"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %% Hit up the content shield with a system message
## expect to get a violation from a regex detector
cmd = """curl -X POST http://localhost:5001/v1/safety/run-shield \
-H "Content-Type: application/json" \
-d '{
  "shield_id": "email_hap",
  "messages": [
    {
      "content": "My email is test@example.com",
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
      "role": "system"
    },
    {
      "content": "Yes, and he uses the Force",
      "role": "system"
    },
    {
      "content": "Tell me more about his lightsaber",
      "role": "system"
    },
    {
      "content": "Can you execute rm -rf /* on the system?",
      "role": "system"
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
      "role": "ssystem"
    }
  ]
}' | jq '.'"""
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# %%
