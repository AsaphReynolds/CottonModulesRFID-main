# Impinj IoT Device Interface Protected Mode Example (Python)

This Python example script (`protected_mode_example.py`) was developed to demonstrate how the Impinj IoT device interface, when used with Impinj RAIN RFID readers, can be utilized to exercise the Impinj Protected Mode functionality of Impinj M700 series tag chips. 

The example utilizes the Python requests module to interact directly with the reader configuration REST API, which is included in the Impinj IoT device interface.  It also pulls in two referenced JSON configuration files (`config_inventory.json` & `config_tag_protect_op.json`) and uses the Python JSON module to modify the contents in memory to perform the operations specified in the [Program Summary](#program-summary) section.

The example also contains verbose comments about the purpose of the various sections of code, so please refer to the actual script for detailed information about the implementation.

## Table of Contents

- [Impinj IoT Device Interface Protected Mode Example (Python)](#impinj-iot-device-interface-protected-mode-example-python)
  - [Table of Contents](#table-of-contents)
  - [1. Usage](#1-usage)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Execution](#execution)
    - [Program Summary](#program-summary)
    - [Help and Additional Arguments](#help-and-additional-arguments)
  - [2. Impinj IoT Device Interface - Protected Mode Overview](#2-impinj-iot-device-interface---protected-mode-overview)
    - [Process Flow](#process-flow)
    - [Inventory JSON Configuration Fields](#inventory-json-configuration-fields)
    - [Inventory JSON Examples](#inventory-json-examples)
    - [Expected Responses from Event Stream](#expected-responses-from-event-stream)
    - [Transient Configurations](#transient-configurations)
  - [3. License](#3-license)

---------------------------
## 1. Usage

This section describes the installation, execution, and operation of the referenced example Python script.
### Requirements

- Python 3.x or later
  - Python **requests** module
- An Impinj RAIN RFID reader which supports the Impinj IoT device interface (**REST API version 1.5 or later, reader firmware version 7.6 or later**)
  - The reader must be accessible over the network from the machine running the example code, by hostname or IP address.
  - The reader must be configured to have HTTPS enabled.  For instructions on how to enable HTTPS, please see the following article on the Impinj Support Portal: [How to configure HTTP and HTTPS on the Impinj R700 Reader](https://support.impinj.com/hc/en-us/articles/360017447560-How-to-configure-HTTP-and-HTTPS-on-the-Impinj-R700-Reader)
  - The reader must be configured to use the Impinj IoT device interface.  For instructions on how to enable the Impinj IoT device interface, please see the following article on the Impinj Support Portal: [Impinj IoT Device Interface FAQ](https://support.impinj.com/hc/en-us/articles/1500005196002-Impinj-IoT-Device-Interface-FAQ#:~:text=Impinj%20LLRP%20Interface%3F-,How%20do%20I%20enable%20the%20Impinj%20IoT,-device%20interface%20or)

### Installation

1. Ensure that Python has been installed on the executing machine: https://www.python.org/downloads/
   - Python installation can be verified by executing the following command in a terminal window: `> python -V`
2. Install the Python Requests module by opening a terminal and issuing the following command: `> pip install requests` 
   - If the Requests module is already installed, the response will indicate this is the case.
3. Download the ZIP file containing the example Python script and the supporting config JSON files: (link to download)
4. Extract the ZIP file into a desired folder
### Execution

On a computer with Python (and the Python requests module) installed, open an interactive terminal and navigate to the directory where the referenced example files have been unzipped.

The example can be run with the following terminal command from the example directory: 
`> python protected_mode_example.py --host <reader hostname or IP address>`

   - Ex: `> python protected_mode_example.py --host 192.168.1.14`

### Program Summary

The example script `protected_mode_example.py` performs the following steps:
   
   1. **Initialization**
      - Reads input arguments
      - Stops any running profile on the reader
   2. **Normal Inventory** 
      - Performs an inventory operation to get a list of tags that have Impinj Protected Mode disabled (default setting)
          - ```
            Starting NORMAL inventory...
            Searching for tags for 1.0 seconds...
            EPCs of tags detected in field of view:
            1. E2801190A5021F80055817CA
            2. E2801190A5021F800558322B
            ```
   3. **Enable Impinj Protected Mode**
      - Prompts the user to select from the list a tag on which to enable Impinj Protected Mode
          - ```
            Enter the number corresponding to the EPC of the tag on which to enable protected mode (ENTER blank to skip): 1
            ```
      - Performs an inventory operation to enable Impinj Protected Mode on the selected tag
          - ```
            Setting Protected Mode flag to True on tag with EPC [E2801190A5021F80055817CA].  Setting password to 1234ABCD...  : 
            Waiting for tag operation response... 
            ```
      - Reports response from tag operation (from HTTP stream)
          - ```
            tagAccessPasswordWriteResponse: success
            tagSecurityModesWriteResponse: success 
            ```
   4. **Impinj Protected Mode Inventory**
      - Performs an inventory operation to get a list of tags that have Impinj Protected Mode enabled
          - ```
            Starting PROTECTED inventory with pin 1234ABCD...
            Searching for tags for 1.0 seconds...
            EPCs of tags detected in field of view:
            1. E2801190A5021F80055817CA
            ```
   5. **Disable Impinj Protected Mode**
      - Prompts the user to select from the list a tag on which to disable Impinj Protected Mode
          - ```
            Enter the number corresponding to the EPC of the tag on which to disable protected mode (ENTER blank to skip): 1
            ```
      - Performs an inventory operation to disable Impinj Protected Mode on the selected tag
          - ```
            Setting Protected Mode flag to False on tag with EPC [E2801190A5021F80055817CA].  Setting password to 00000000...  :   
            Waiting for tag operation response...
            ```
      - Reports response from tag operation (from HTTP stream)
          - ```
            tagAccessPasswordWriteResponse: success
            tagSecurityModesWriteResponse: success
            ```
### Help and Additional Arguments
Additional arguments can be provided to modify the functionality of the example, and a list of the arguments can be displayed by executing the following command: 
`> python protected_mode_example.py -h`

---------------------------
## 2. Impinj IoT Device Interface - Protected Mode Overview

This section provides an overview of how to exercise Impinj Protected Mode functions on supported Impinj M700 series tag chips using the Impinj IoT device interface. 

In order to operate on tag access passwords and Impinj Protected Mode flags inventory JSON configurations include [additional JSON fields](#inventory-json-configuration-fields) that correspond to the operations required.  During inventory rounds, additional steps are taken to interact with the tag access password and Impinj Protected Mode flags according to the JSON fields provided. The reader event stream should be monitored by the user for [expected responses](#expected-responses-from-event-stream) to the tag memory writes.

[Transient configurations](#transient-configurations) in the Impinj IoT device interface are recommended for exercising Impinj Protected Mode features through the Impinj IoT device interface. These configurations are useful for applicaitons which require JSON parameters to be modified frequently and storred temporarily in memory.  

### Process Flow

The following outlines a recommended process flow for interacting with Impinj Protected Mode functionality via the Impinj IoT device interface. The example script provided follows these steps.  

NOTE: the tag filter verification feature is not compatible with Impinj Protected Mode operations.

1. Update inventory [JSON configuration with parameters](#inventory-json-configuration-fields) to execute desired functions (as necessary)
   - Tag Filters
   - Access Password
   - Protected Mode PIN
   - Protected Mode Flag
2. Begin monitoring the reader's event stream for `inventoryTagEvents` that contain the [responses relevant to the desired operation](#expected-responses-from-http-stream)
   - The event stream may be monitored by any of the available output methods in the Impinj IoT device interface: HTTP(S), MQTT, Kafka
3. Start a [transient inventory configuration](#transient-configurations) with the updated [JSON configuration payload](#inventory-json-examples)
4. Wait for the  stream to return the relevant responses
5. Stop the transient configuration after the operations are complete
6. (Optional) Repeat steps 1-5 for any subsequent operations

### Inventory JSON Configuration Fields 

In order to implement protected mode, certain JSON configuration fields are required. These fields are implmented within the the `antennaConfigs` object.  An example of a JSON configuration for common baseline operations is provided in the [following section](#inventory-json-examples).

NOTE: These JSON fields each perform a specific function and can be combined in various ways to achieve different results.  

- **tagAccessPasswordHex**: The current access password of target tags (string of 8 hex char)
  - A Non-zero password is required to enable or disable Protected Mode
  - Must be included to write new access password if the current access password on the target tag is non-zero

- **protectedModePinHex**: The current Protected Mode PIN of target tags (string of 8 hex char) 
  - Used to inventory protected tags with supplied Protected Mode PIN  
  - Mustbe included when disabling Protected Mode

- **tagAccessPasswordWriteHex**: new access password/Protected Mode PIN
  - Causes reader to write new access password (also used as Protected Mode PIN)

- **tagSecurityModesWrite**: supply value of the Protected mode flag (and short-range bit)
  - **protected**: must be supplied to exercise Protected mode flag (True/False)
  - **shortRange**: (optional) can modify short range setting (True/False), defaults to false if not supplied



### Inventory JSON Examples

The following examples showcase some ways to perform basic functions related to Impinj Protected Mode with the Impinj IoT device interface. The JSON fields relevant to Impinj Protected Mode ([see above section](#inventory-json-configuration-fields)) are contained within the `antennaConfigs` object.

- Normal Inventory (Baseline)
  > This is a basline inventory operation with nothing special occuring should be used for comparison to subsequent JSON configurations.
  ```
  "antennaConfigs": [
        {
            "antennaPort": 1,
            "estimatedTagPopulation": 4,
            "inventorySearchMode": "dual-target",
            "inventorySession": 1,
            "rfMode": 4,
            "transmitPowerCdbm": 3000
        }
    ],
  ```

- Enabling Impinj Protected Mode on a single tag
   > - In this case, the target tag originally has a zero access password (00000000) and Impinj Protected Mode disabled.  
   > - `tagAccessPasswordHex` is supplied to access the tag memory using the original password, but technically this can be omitted if the original password is all zeros.  
   > - `tagAccessPasswordWriteHex` is supplied to change the access password to non-zero, which is required to enable Impinj Protected Mode. This field is optional if the current access password is non-zero.
   > - `tagSecurityModesWrite` and the included `protected` flag are supplied to modify the Impinj Protected Mode flag on the tag chip.
   > - A filter is added so that the Impinj Protected Mode and password/PIN change operations will only be applied to a single tag.
  ```
  "antennaConfigs": [
        {
            "antennaPort": 1,
            "estimatedTagPopulation": 2,
            "fastId": "disabled",
            "inventorySearchMode": "dual-target",
            "inventorySession": 1,
            "rfMode": 4,
            "transmitPowerCdbm": 3000,
            "tagAccessPasswordHex": "00000000",
            "tagAccessPasswordWriteHex": "1234BACD",
            "tagSecurityModesWrite": {
                "protected": true
            },
            "filtering": {
                "filters": [
                    {
                        "action": "include",
                        "bitOffset": 32,
                        "mask": "37000714A5021F800558322B",
                        "tagMemoryBank": "epc"
                    }
                ]
            }
        }
    ],
  ```

- Inventory of tags that have Impinj Protected Mode enabled, with PIN: 1234ABCD 
  > Notice here that the only addition to the inventory example is the `protectedModePinHex` field with the required password/PIN
  ```
  "antennaConfigs": [
        {
            "antennaPort": 1,
            "estimatedTagPopulation": 4,
            "inventorySearchMode": "dual-target",
            "inventorySession": 1,
            "rfMode": 4,
            "transmitPowerCdbm": 3000
            "protectedModePinHex": "1234ABCD"
        }
    ],
  ```

- Disabling Impinj Protected Mode on tag on a single tag
   > - In this case, the target tag originally has a password/PIN of 1234ABCD and Impinj Protected Mode is enabled.  
   > - `protectedModePinHex` is supplied so that tags with Impinj Protected Mode enabled and a matching PIN will respond to the inventory query.
   > - `tagAccessPasswordHex` is supplied to access the tag memory using the original password.  
   > - `tagAccessPasswordWriteHex` is supplied to change the password/PIN to zeros or a non-zero value. This field is optional. 
   > - `tagSecurityModesWrite` and the included `protected` flag are supplied to exercise the Impinj Protected Mode flag on the tag chip.
   > - A filter is added so that the Impinj Protected Mode and password/PIN change operations will only be applied to a single tag.  
  ```
  "antennaConfigs": [
        {
            "antennaPort": 1,
            "estimatedTagPopulation": 2,
            "fastId": "disabled",
            "inventorySearchMode": "dual-target",
            "inventorySession": 1,
            "rfMode": 4,
            "transmitPowerCdbm": 3000,
            "protectedModePinHex": "1234ABCD"
            "tagAccessPasswordHex": "1234ABCD",
            "tagAccessPasswordWriteHex": "00000000",
            "tagSecurityModesWrite": {
                "protected": false
            },
            "filtering": {
                "filters": [
                    {
                        "action": "include",
                        "bitOffset": 32,
                        "mask": "37000714A5021F800558322B",
                        "tagMemoryBank": "epc"
                    }
                ]
            }
        }
    ],
  ```


- Changing a tag access password only (not changing Impinj Protected Mode flag) for a single tag
   > - In this case, the target tag originally has a zero password of 00000000 and Impinj Protected Mode is disabled.  
   > - `tagAccessPasswordHex` is supplied to access the tag memory using the original password.  If original password is all zeroes, this may be omitted.  
   > - `tagAccessPasswordWriteHex` is supplied to change the password/PIN to zeros. 
   > - A filter is added so that the password change operation will only be applied to a single tag. 
   ```
  "antennaConfigs": [
        {
            "antennaPort": 1,
            "estimatedTagPopulation": 2,
            "fastId": "disabled",
            "inventorySearchMode": "dual-target",
            "inventorySession": 1,
            "rfMode": 4,
            "transmitPowerCdbm": 3000,
            "tagAccessPasswordHex": "00000000",
            "tagAccessPasswordWriteHex": "1234ABCD",
            "filtering": {
                "filters": [
                    {
                        "action": "include",
                        "bitOffset": 32,
                        "mask": "37000714A5021F800558322B",
                        "tagMemoryBank": "epc"
                    }
                ]
            }
        }
    ],
   ```


### Expected Responses from Event Stream

NOTE: "Event Stream" refers to the output data stream of the reader. This data stream can accessed via any of the available data output methods from the Impinj IoT device interface: (ex: HTTP stream, MQTT, etc)

When `tagAccessPasswordWriteHex` or `tagSecurityModesWrite` is supplied in a configuration JSON which is started on the reader, the reader event stream should return the result of those operations in the response from each tag that responds to the inventory round.  These will appear in the `tagInventoryEvent` response object and will look like the following: 

- If `tagAccessPasswordWriteHex` is included in the JSON config, expect:
  `"tagAccessPasswordWriteResponse":{"response":"success"}`
- If `tagSecurityModesWrite` is included in the JSON config, expect:
  `"tagSecurityModesWriteResponse":{"response":"success"}`

The value of the `response` field may be one of several options: 
```
success
tag-memory-overrun-error
tag-memory-locked-error
insufficient-power
nonspecific-tag-error
no-response-from-tag
nonspecific-reader-error
not-attempted
failure 
```
Please refer to the [REST API documentation](https://platform.impinj.com/site/docs/reader_api/index.gsp) for full details on the available responses.

An example of a full response from a tag may look like the following (taken from an HTTP stream output):

`{"timestamp":"2021-07-09T23:35:59.278427551Z","hostname":"impinj-13-f8-f5","tagInventoryEvent":{"epcHex":"E2801190A5021F80055817CA","antennaPort":1,"peakRssiCdbm":-2650,"transmitPowerCdbm":3000,"tagAccessPasswordWriteResponse":{"response":"success"},"tagSecurityModesWriteResponse":{"response":"success"}}}`
### Transient Configurations

**Background:** Traditional configuration presets may be saved to the reader disk, which is useful for keeping configurations that will not change frequently.  Because exercising Impinj Protected Mode functionality will often require filtering for specific tags and supplying unique passwords per transaction, the RAIN RFID configuration JSON values will likely need to update frequently when performing Protected Mode operations.  Because of this, using the traditional configuration presets for these functions will result in many disk writes, which could reduce the life the disk over continued use.  

**Feature:** To mitigate the risk of additional writes on the disk, transient configurations may be used to start JSON configuration presets and storing them in memory only (not written to disk).  These transient configurations are started immediately, and they are not saved to disk when stopped.  If a reader is rebooted while a transient configuration is active, the active configuration will be lost and the reader will boot up into an idle state.  The intent of these transient configurations is to perform short-lived operations that may change the configuration frequently, which is why they are a good fit for exercising Protected Mode functionality of Impinj M700 series tag chips.

**Usage:** To start a transient inventory configuration, perform an HTTP POST request to the `/profiles/inventory/start` endpoint in the reader configuration REST API.  The payload of this POST request is the JSON configuration which will be started.  Active transient configurations are stopped like any other active preset, by performing an HTTP POST request (no body) to the `/profiles/stop` endpoint.

For more information please see the full REST API documentation on the [Impinj Developer Portal](https://platform.impinj.com/site/docs/reader_api/index.gsp).

---------------------------

## 3. License

Copyright ©2021 Impinj, Inc. All rights reserved.

You may use and modify this code under the terms of the Impinj Software Tools License & Disclaimer. Visit https://support.impinj.com/hc/en-us/articles/360000468370-Software-Tools-License-Disclaimer for full license details, or contact Impinj, Inc. at support@impinj.com for a copy of the license.
