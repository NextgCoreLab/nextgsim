// MongoDB script to add a test subscriber for nextgsim
// Run with: docker exec -i nextgcore-mongodb mongosh open5gs < add-subscriber.js
//
// Subscriber: IMSI 999700000000001
// K:  465B5CE8B199B49FAA5F0A2EE238A6BC
// OPc: E8ED289DEBA952E4283B54E88E6D834D

// Switch to the open5gs database
db = db.getSiblingDB('open5gs');

// Remove existing subscriber if present
db.subscribers.deleteOne({ "imsi": "999700000000001" });

// Insert the test subscriber
db.subscribers.insertOne({
  "imsi": "999700000000001",
  "msisdn": [],
  "imeisv": "4370816125816151",
  "mme_host": [],
  "mme_realm": [],
  "purge_flag": [],
  "security": {
    "k": "465B5CE8B199B49FAA5F0A2EE238A6BC",
    "opc": "E8ED289DEBA952E4283B54E88E6D834D",
    "amf": "8000",
    "sqn": NumberLong(513)
  },
  "ambr": {
    "downlink": { "value": 1, "unit": 3 },
    "uplink": { "value": 1, "unit": 3 }
  },
  "slice": [
    {
      "sst": 1,
      "default_indicator": true,
      "session": [
        {
          "name": "internet",
          "type": 3,
          "pcc_rule": [],
          "ambr": {
            "downlink": { "value": 1, "unit": 3 },
            "uplink": { "value": 1, "unit": 3 }
          },
          "qos": {
            "index": 9,
            "arp": {
              "priority_level": 8,
              "pre_emption_capability": 1,
              "pre_emption_vulnerability": 1
            }
          }
        }
      ]
    }
  ],
  "access_restriction_data": 32,
  "subscriber_status": 0,
  "network_access_mode": 0,
  "__v": 0
});

print("Subscriber 999700000000001 added successfully!");

// Verify the insertion
const sub = db.subscribers.findOne({ "imsi": "999700000000001" });
print("Verified subscriber:");
printjson(sub);
