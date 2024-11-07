package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.Command;
import org.example.processoremulatorfx.Models.ProgramModel;

import java.util.Objects;

public class AddCommandViewController {
  ProgramModel programModel = BProgramModel.build();

  @FXML
  void add() throws Exception {
    if (ComboBoxInstruction.getValue() != null) {
      if (((ComboBoxInstruction.getValue().equals("init") || ComboBoxInstruction.getValue().equals("ld")
              || ComboBoxInstruction.getValue().equals("st") || ComboBoxInstruction.getValue().equals("mv"))
              && (!Objects.equals(Arg1.getText(), "") && !Objects.equals(Arg2.getText(), ""))) ||
              ((ComboBoxInstruction.getValue().equals("print") || ComboBoxInstruction.getValue().equals("add")
              || ComboBoxInstruction.getValue().equals("sub") || ComboBoxInstruction.getValue().equals("mult")
              || ComboBoxInstruction.getValue().equals("div")))) {
        programModel.addCommand(new Command((String) ComboBoxInstruction.getValue(), Arg1.getText(), Arg2.getText()));
      }
      else
        System.out.println("Warning: Enter the parameters");
    }
    else
      System.out.println("Warning: Choose command");
  }

  @FXML
  ComboBox ComboBoxInstruction;
  @FXML
  TextField Arg1;
  @FXML
  TextField Arg2;
  @FXML
  Button AddButton;

  @FXML
  void initialize()
  {
    ComboBoxInstruction.getItems().addAll("init", "ld", "st", "mv", "print", "add", "sub", "mult", "div");
    Arg1.setDisable(true);
    Arg2.setDisable(true);

    ComboBoxInstruction.setOnAction(event -> {
      String selectedCommand = (String) ComboBoxInstruction.getValue();
      Arg1.setText("");
      Arg2.setText("");

      if (selectedCommand.equals("print") || selectedCommand.equals("add") || selectedCommand.equals("sub") ||
              selectedCommand.equals("div") || selectedCommand.equals("mult")) {
        Arg1.setDisable(true);
        Arg2.setDisable(true);
      } else {
        Arg1.setDisable(false);
        Arg2.setDisable(false);
//        if(selectedCommand.equals("init")) {
//          Arg1.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("\\d*")) { // \\d* означает "любое количество цифр"
//              Arg1.setText(newValue.replaceAll("\\D", "")); // Удаляет все, кроме цифр
//            }
//          });
//          Arg2.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("\\d*")) { // \\d* означает "любое количество цифр"
//              Arg2.setText(newValue.replaceAll("\\D", "")); // Удаляет все, кроме цифр
//            }
//          });
//        }else if (selectedCommand.equals("ld") || selectedCommand.equals("st")) {
//          Arg1.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("[a-d]*")) {
//              Arg1.setText(newValue.replaceAll("[^a-d]", ""));
//            }
//          });
//          Arg2.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("\\d*")) { // \\d* означает "любое количество цифр"
//              Arg2.setText(newValue.replaceAll("\\D", "")); // Удаляет все, кроме цифр
//            }
//          });
//        } else if (selectedCommand.equals("mv")) {
//          Arg1.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("[a-d]*")) {
//              Arg1.setText(newValue.replaceAll("[^a-d]", ""));
//            }
//          });
//          Arg2.textProperty().addListener((observable, oldValue, newValue) -> {
//            if (!newValue.matches("[a-d]*")) {
//              Arg2.setText(newValue.replaceAll("[^a-d]", ""));
//            }
//          });
//        }
      }
    });
  }
}
