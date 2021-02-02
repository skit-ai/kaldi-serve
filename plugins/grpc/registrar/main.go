package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// It will try to register the service into the local Consul agent 10 times, after that it will stop.
// A simple PUT requests to the /v1/agent/service/register of Consul, passing the correct payload.
// To run the script, you need to set the following ENV vars:
//
// export APP_NAME=fake_service
// export APP_PORT=5001
// export CONSUL_PORT=8500
// export TAGS=svc1
// go run register/main.go
//
// Description for the env vars are as follows:
//
// APP_NAME: Name of the app being registered
// APP_PORT: Port on which the app is running
// CONSUL_PORT: Port on which consul is running. (Usually 8500)
// TAGS: Tags to be used to signify the app
//
// Additionally, there are 2 optional env vars where you can specify the health check API details for the service
// HEALTH_CHECK_TYPE: Can be "http" or "grpc"
// HEALTH_CHECK_ENDPOINT: API endpoint for the health check.
// Documentation for the same can be found here: https://www.consul.io/api/agent/check#register-check
func main() {
	tags := os.Getenv("TAGS")
	if tags == "" {
		logFatal("tags")
	}

	name := os.Getenv("APP_NAME")
	if tags == "" {
		logFatal("tags")
	}

	_port := os.Getenv("APP_PORT")
	if _port == "" {
		logFatal("app port")
	}
	port, err := strconv.Atoi(_port)
	if err != nil {
		log.Fatalf("Unable to convert \"%s\" to int: %v", _port, err)
	}

	consulPort := os.Getenv("CONSUL_PORT")
	if consulPort == "" {
		logFatal("consul port")
	}

	// Can be either HTTP or GRPC
	healthCheckProtocol := os.Getenv("HEALTH_CHECK_TYPE")
	healthCheckEndpoint := os.Getenv("HEALTH_CHECK_ENDPOINT")



	//Encode the data
	postBody := map[string]interface{}{
		"name":    name,
		"tags":    strings.Split(tags, ","),
		"address": "",
		"port":    port,
	}

	// Adding a health check endpoint if specified in the env vars
	if healthCheckEndpoint != "" && (healthCheckProtocol == "http" || healthCheckProtocol == "grpc") {
		postBody["checks"] = []map[string]string{
			{"http": fmt.Sprintf("http://localhost:%s/hostname", consulPort), "interval": "5s"},
		}
	}

	consulRegisterEndpoint := fmt.Sprintf("http://localhost:%s/v1/agent/service/register", consulPort)

	// Attempt to register the service 10 times before giving up
	for i := 0; i < 10; i++ {
		if err = registerService(consulRegisterEndpoint, postBody); err != nil {
			// Wait for a second and then attempt tp register
			time.Sleep(1 * time.Second)
			continue
		} else {
			break
		}
	}

	if err != nil {
		log.Fatalf("Unable to register to consul: %v", err)
	}
}

// Simply PUT request to consul's register endpoint to register a service
func registerService(consulRegisterEndpoint string, postBody map[string]interface{}) error {
	var b []byte
	var err error
	if b, err = json.Marshal(postBody); err != nil {
		return err
	}
	responseBody := bytes.NewBuffer(b)
	req, err := http.NewRequest("PUT", consulRegisterEndpoint, responseBody)
	//Handle Error
	if err != nil {
		return err
	}

	httpClient := &http.Client{
		Timeout: time.Duration(21 * time.Second),
	}

	req.Header.Set("Accept", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	sb := string(body)

	if resp.StatusCode != 200 {
		return errors.New(fmt.Sprintf("Response code %d from consul. Body: \"%s\"", resp.StatusCode, sb))
	} else {
		fmt.Println("Service registered with consul successfully !")
	}

	return nil
}

func logFatal(varName string) {
	if name, err := os.Hostname(); err != nil {
		log.Fatalf("No %v specified for host %v", varName, name)
	} else {
		log.Fatalf("No %v specified for this host", varName)
	}
}
