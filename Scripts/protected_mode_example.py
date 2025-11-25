#!/usr/bin/env python

import argparse
import json
from json import decoder\

import signal
import sys

import requests
from requests.exceptions import Timeout
from requests.models import HTTPBasicAuth

import time
import threading

# signalHandler: called to terminate the program when CTRL+C is pressed
def signalHandler(signal, frame):
    sys.exit(0)

# getCommandLineArguments: parses the input arguments from the command line and returns them to the calling function
def getCommandLineArguments():

    parser = argparse.ArgumentParser(description='This is an example program showing how to enable and disable Impinj Protected Mode with the Impinj IoT device interface.')
    parser.add_argument('--host', '-ip', action='store', dest='host', required=True, help='(string) IP address or hostname of the reader supporting the Impinj IoT device interface')
    parser.add_argument('--currentAccessPass', '-cp', action='store', dest='currentAccessPass', required=False, default='00000000', help='(8x hexadecimal char) current access used to read unprotected tags, will be set again when protected mode is disabled')
    parser.add_argument('--newAccessPass', '-np', action='store', dest='newAccessPass', required=False, default='1234ABCD', help='(8x hexadecimal char) new access password to set when protected mode is enabled')
    parser.add_argument('--antenna', '-ant', action='store', dest='antenna', required=False, default=1, help='(integer) reader antenna to use for example RFID operations')
    parser.add_argument('--initialSearchTimeSec', '-tsearch', action='store', dest='initialSearchTimeSec', required=False, default=1, help='(float) time in seconds seconds to search for the initial list of tag EPCs')
    return parser.parse_args()

# getTagList: subscribes to the reader's HTTP stream endpoint and looks for unique tag responses, which are built into a list.  The list of unique tag EPCs is returned to the calling function.
def getTagList(endpointDataStream, timeout):
    
    try: 
        response = requests.get(endpointDataStream, stream=True, verify=False, timeout=timeout) # Connect to the reader's HTTP event stream

        initialTime = time.time()                                                               # initialize the search timer

        tagList = []                                                                            # initialize the tag list
        print ('Searching for tags for {} seconds...'.format(timeout))
        print ('EPCs of tags detected in field of view:')
        for event in response.iter_lines():                                                     # iterate through each line returned from stream
            string = event.decode("utf8")                                                       # decode the raw bit stream into a string
            if(len(string) > 0):
                jsonEvent = json.loads(string)                                                  # parse string into JSON dictionary
                if 'tagInventoryEvent' in jsonEvent:                                            # search for "tagInventoryEvent" field
                    epcHex = jsonEvent['tagInventoryEvent']['epcHex']
                    try:                                                                        # if tag EPC already exists in tag list, do nothing
                        tagList.index(epcHex)
                    except ValueError as ve:                                                    # if tag EPC does not exist in tag list, add EPC to tag list
                        tagList.append(epcHex)
                        print('{}. {}'.format(tagList.index(epcHex)+1, epcHex))
            if (time.time() - initialTime >= timeout):                                          # when time has elapsed, exit loop and stop looking for new tags
                break

    except requests.exceptions.RequestException as err:
        print('No tags detected in field of view.')

    return tagList                                                                              # return built tag list to calling function

# checkTagOpResponse: subscribes to the HTTP stream and specifically looks for tag responses that include 'tagAccessPasswordWriteResponse' and 'tagSecurityModesWriteResponse'
def checkTagOpResponse(endpointDataStream, timeout, epcHex):
    
    try: 
        response = requests.get(endpointDataStream, stream=True, verify=False, timeout=timeout) # Connect to the reader's HTTP event stream

        print ('Waiting for tag operation response...'.format(timeout))
        for event in response.iter_lines():        
            string = event.decode("utf8")                                                       # decode the raw bit stream into a string
            if(len(string) > 0):
                jsonEvent = json.loads(string)                                                  # parse the JSON string into a dictionary
                if 'tagInventoryEvent' in jsonEvent:                                            # look for specific events with the desired tag responses
                    if jsonEvent['tagInventoryEvent'].keys() >= {'tagAccessPasswordWriteResponse', 'tagSecurityModesWriteResponse'}:
                        print('tagAccessPasswordWriteResponse: {}'.format(jsonEvent['tagInventoryEvent']['tagAccessPasswordWriteResponse']['response']))
                        print('tagSecurityModesWriteResponse: {}'.format(jsonEvent['tagInventoryEvent']['tagSecurityModesWriteResponse']['response']))  
                        break                                                                   # after one matching event is detected, exit loop

    except requests.exceptions.RequestException as err:    # if no events contain the proper tag response in the specified timeout, throw an error.
        print('ERROR: no response seen from tag before timeout ({} sec).'.format(timeout))

# operationTagProtect: loads the config_tag_protect_op.json file and modifies the contents in memory to perform the requested protected mode operation (enable or disable) by running a transient configuration on the reader.  The function also spawns a second thread to look at the reader's HTTP stream for operation responses from the tag.
def operationTagProtect(protectedModeFlag, epcHex, endpointDataStream, endpointTransientConfigStart, programArguments):
    # Update config_tag_protect_op JSON with function argument values
    with open('./config_tag_protect_op.json') as fileConfigLoad:            # load the config_tag_protect_op.json file into memory
        configTagProtectOp = json.load(fileConfigLoad)                      # parse the JSON file into object dictionary for modification

    if (protectedModeFlag):                                                 # if enabling protected mode, old password will be used for access, new pass will be written
        tagAccessPasswordHex = programArguments.currentAccessPass
        protectedModePinHex = programArguments.newAccessPass
        tagAccessPasswordWriteHex = programArguments.newAccessPass
    else:                                                                   # if disabling protected mode, new password will be used for access, old pass will be written
        tagAccessPasswordHex = programArguments.newAccessPass
        protectedModePinHex = programArguments.newAccessPass
        tagAccessPasswordWriteHex = programArguments.currentAccessPass

    for item in configTagProtectOp['antennaConfigs']:                       # updating JSON dictionary for transient configuration with specified values
        item ['antennaPort'] = programArguments.antenna
        item ['filtering']['filters'][0]['mask'] = epcHex
        item ['tagAccessPasswordHex'] = tagAccessPasswordHex
        item ['tagSecurityModesWrite']['protected'] = protectedModeFlag  # set to specified T/F value
        item ['tagAccessPasswordWriteHex'] = tagAccessPasswordWriteHex
        if (protectedModeFlag):                                              # if enabling protected mode, do not need 'protectedModePinHex'
            continue
        else:                                                                # if disabling protected mode, add 'protecteModePinHex' with current PIN
            item ['protectedModePinHex'] = protectedModePinHex
            
        

    # Spawn thread to read HTTP stream and monitor for protect operation response from tag
    threadCheckTagOpResponse = threading.Thread(target = checkTagOpResponse, args=(endpointDataStream, 3, epcHex), daemon=True)
    threadCheckTagOpResponse.start()   

    # Start transient configuration to put selected tag into Protected Mode
    response = requests.post(endpointTransientConfigStart, auth=HTTPBasicAuth('root', 'impinj'), json = configTagProtectOp, verify=False )
    print("Setting Protected Mode flag to {} on tag with EPC [{}].  Setting password to {}... ".format(protectedModeFlag, epcHex, tagAccessPasswordWriteHex), ": ", response.text, '\n')

    # Join thread which was spawned to read HTTP stream data
    threadCheckTagOpResponse.join()

# userInputSelectTag: prompts the user to select a tag from the detected tag list
def userInputSelectTag(enableFlag, tagList):
    inputVerify = True
    selectedTagEpcHex='default'
    if enableFlag:
        enableText = "enable"
    else:
        enableText = "disable"

    while (inputVerify):
        tagSelection = int(input('Enter the number corresponding to the EPC of the tag on which to {} protected mode (ENTER blank to skip): '.format(enableText)))
        try: 
            selectedTagEpcHex = tagList[tagSelection-1]
            inputVerify = False
        except:
            print('Specified value is invalid.  Please select a number from the list of detected tags.')
            inputVerify = True

    return selectedTagEpcHex

# main: the main program that calls the other functions
def main():
    # Handle Ctrl+C interrupt
    signal.signal(signal.SIGINT, signalHandler)

    arguments = getCommandLineArguments()  # get command line arguements

    # use command line arguments to build variables
    readerHostname = '{}'.format(arguments.host)

    urlBase = 'https://{}/api/v1'.format(readerHostname)
    print('\nBase URL: {}\n'.format(urlBase))

    # build endpoint URL variables
    endpointTransientConfigStart = '{}/profiles/inventory/start'.format(urlBase)
    endpointProfileStop = '{}/profiles/stop'.format(urlBase)
    endpointDataStream = '{}/data/stream'.format(urlBase)

    # Begin HTTP requests to reader
    requests.packages.urllib3.disable_warnings() # suppress the warnings about accepting self-signed certificate on the reader

    # Stop the active profile, this is the first HTTPS call to the reader, so check that it succeeds, and if not, print error message.
    try:
        response = requests.post(endpointProfileStop, auth=HTTPBasicAuth('root', 'impinj'), verify=False)
         # print("   Stopping active profile...", response.url, ": ", response.text)
    except:
        print("Failed to communicate with the reader ({}).  Make sure it is accessible on the network, that the Impinj IoT device interface is enabled, and that HTTPS is enabled.".format(arguments.host))
        exit()

    # Update config_inventory JSON with function argument values
    with open('./config_inventory.json') as fileConfigLoad:
        configInventoryNormal = json.load(fileConfigLoad)

    for item in configInventoryNormal['antennaConfigs']:
        item ['antennaPort'] = arguments.antenna

    # print('configInventoryNormal: ', configInventoryNormal, '\n')

    print('\nStarting NORMAL inventory...')

    # Start the normal tag query config and get the list of tags
    response = requests.post(endpointTransientConfigStart, auth=HTTPBasicAuth('root', 'impinj'), json = configInventoryNormal, verify=False )
    tagList = getTagList(endpointDataStream, float(arguments.initialSearchTimeSec))

    # Stop the active profile
    response = requests.post(endpointProfileStop, auth=HTTPBasicAuth('root', 'impinj'), verify=False)

    if (len(tagList) > 0):
        # Prompt user to select a tag to protect
        try: 
            selectedTagEpcHex = userInputSelectTag(True, tagList)
        except:
            print('No tags selected for Protected Mode enable.')
        else:
            operationTagProtect(True, selectedTagEpcHex, endpointDataStream, endpointTransientConfigStart, arguments)

    # Stop the active profile
    response = requests.post(endpointProfileStop, auth=HTTPBasicAuth('root', 'impinj'), verify=False)

    # Update config_inventory JSON with function argument values to enable inventory of PROTECTED tags
    with open('./config_inventory.json') as fileConfigLoad:
        configInventoryProtected = json.load(fileConfigLoad)

    for item in configInventoryProtected['antennaConfigs']:
        item ['antennaPort'] = arguments.antenna
        item ['protectedModePinHex'] = arguments.newAccessPass

    # print('configInventoryProtected: ', configInventoryProtected, '\n')

    print('\nStarting PROTECTED inventory with pin {}...'.format(arguments.newAccessPass))

    # Start the protected tag query transient configuration and get the list of tags
    response = requests.post(endpointTransientConfigStart, auth=HTTPBasicAuth('root', 'impinj'), json = configInventoryProtected, verify=False )
    tagList = getTagList(endpointDataStream, float(arguments.initialSearchTimeSec))

    # Stop the active profile
    response = requests.post(endpointProfileStop, auth=HTTPBasicAuth('root', 'impinj'), verify=False)

    if (len(tagList) > 0):
        # Prompt user to select a tag on which to disable protected mode
        try:
            selectedTagEpcHex = userInputSelectTag(False, tagList)
        except:
            print('No tags selected for Protected Mode disable.')
        else:
            operationTagProtect(False, selectedTagEpcHex, endpointDataStream, endpointTransientConfigStart, arguments)

    # Stop the active profile
    response = requests.post(endpointProfileStop, auth=HTTPBasicAuth('root', 'impinj'), verify=False)
    # print("   Stopping active profile...", response.url, ": ", response.text)

if __name__ == "__main__":
    main()
