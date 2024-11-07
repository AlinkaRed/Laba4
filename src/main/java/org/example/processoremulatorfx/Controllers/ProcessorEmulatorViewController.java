package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import org.example.processoremulatorfx.BModels.BExecuterModel;
import org.example.processoremulatorfx.Models.ExecuterModel;

public class ProcessorEmulatorViewController {
  ExecuterModel executerModel = BExecuterModel.build();

  @FXML
  void  start() {
    try {
      executerModel.run();
    } catch (Exception e) {
      executerModel.addErr();
      System.out.println("Error: " + e.getMessage());
    }
  }

  @FXML
  void debug() {
    try {
      executerModel.runDebug();
    } catch (Exception e) {
      System.out.println("Warning: " + e.getMessage());
    }
  }

  @FXML
  void reset(){
    executerModel.reset();
  }
}
