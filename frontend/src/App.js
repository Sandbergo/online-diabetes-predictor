import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';

class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      formData: {
        Pregnancies: 0.0,
        Glucose: 0.0,
        BloodPressure: 0.0,
        SkinThickness: 0.0,
        Insulin: 0.0,
        BMI: 0.0,
        DiabetesPedigreeFunction: 0.0,
        Age: 0.0
      },
      result: ""
    };
  }

  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData
    });
  }

  handlePredictClick = (event) => {
    const formData = this.state.formData;
    this.setState({ isLoading: true });
    fetch('http://127.0.0.1:5000/prediction/', 
      {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify(formData)
      })
      .then(response => response.json())
      .then(response => {
        this.setState({
          result: response.result,
          isLoading: false
        });
      });
  }

  handleCancelClick = (event) => {
    this.setState({ result: "" });
  }

  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    const result = this.state.result;

    return (
      <Container>
        <div>
          <h1 className="title">Online Diabetes Predictor</h1>
        </div>
        <div className="content">
          // Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
          <Form>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Pregnancies</Form.Label>
                <Form.Control 
                  value={formData.Pregnancies}
                  name="Pregnancies"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>Glucose</Form.Label>
                <Form.Control 
                  value={formData.Glucose}
                  name="Glucose"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>Blood pressure</Form.Label>
                <Form.Control 
                  value={formData.BloodPressure}
                  name="BloodPressure"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>Skin thickness</Form.Label>
                <Form.Control 
                  value={formData.SkinThickness}
                  name="SkinThickness"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Insulin</Form.Label>
                <Form.Control 
                  value={formData.Insulin}
                  name="Insulin"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>BMI</Form.Label>
                <Form.Control 
                  value={formData.BMI}
                  name="BMI"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>DiabetesPedigreeFunction</Form.Label>
                <Form.Control 
                  value={formData.DiabetesPedigreeFunction}
                  name="DiabetesPedigreeFunction"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>Age</Form.Label>
                <Form.Control 
                  value={formData.Age}
                  name="Age"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Row>
              <Col>
                <Button
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}>
                  { isLoading ? 'Making prediction' : 'Predict' }
                </Button>
              </Col>
              <Col>
                <Button
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}>
                  Reset prediction
                </Button>
              </Col>
            </Row>
          </Form>
          {result === "" ? null :
            (<Row>
              <Col className="result-container">
                <h5 id="result">{result}</h5>
              </Col>
            </Row>)
          }
        </div>
      </Container>
    );
  }
}

export default App;